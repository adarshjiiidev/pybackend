"""
WebResearchAgent — Autonomous Multi-Step Web Research + KB Learning
====================================================================
Phase 2 · Swarm Agent

This agent autonomously:
  1. Plans a research strategy based on the query
  2. Executes multi-round web searches (up to MAX_ROUNDS)
  3. Evaluates each result for relevance and credibility
  4. Decides what follow-up searches to run
  5. Deduplicates and synthesises all findings
  6. Stores new knowledge into the Qdrant KB (learns permanently)
  7. Returns a structured research report

Key capabilities:
  - Perplexity-style deep research: breaks query → sub-questions → searches each
  - Self-directed: LLM decides next search based on gaps in current knowledge
  - Source credibility scoring (financial regulators, exchanges, Reuters > blogs)
  - Knowledge persistence: stores new facts in KB so future queries benefit
  - Parallel sub-searches within each round
  - Deduplication via semantic similarity (embedding cosine distance)
  - Citation tracking: every claim linked to source URL

Input payload keys:
  query           (str, required)  The research question
  depth           (int, 1-5)       Research depth (rounds). Default: 3
  domain_focus    (str)            "indian_markets" | "global" | "crypto" | "general"
  store_in_kb     (bool)           Persist findings to KB. Default: True
  max_results     (int)            Max sources to use. Default: 15

Output AgentResult.data keys:
  query, findings, sources, knowledge_graph, stored_in_kb,
  confidence, research_rounds, follow_up_questions, report_md
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from ..base_agent import AgentResult, AgentStatus, BaseSwarmAgent, SwarmMessage

logger = logging.getLogger(__name__)

AGENT_TYPE = "web_research"

# ---------------------------------------------------------------------------
# Source credibility scoring
# ---------------------------------------------------------------------------

# Higher score = more credible source
_CREDIBILITY_SCORES: Dict[str, float] = {
    # Indian regulatory / official
    "nseindia.com": 0.97,
    "bseindia.com": 0.97,
    "sebi.gov.in": 0.98,
    "rbi.org.in": 0.98,
    "mca.gov.in": 0.90,
    "pib.gov.in": 0.92,
    "finmin.nic.in": 0.92,
    "amfiindia.com": 0.92,
    # Global financial official
    "sec.gov": 0.97,
    "federalreserve.gov": 0.97,
    "imf.org": 0.95,
    "worldbank.org": 0.95,
    "bis.org": 0.95,
    # Top financial news
    "reuters.com": 0.92,
    "bloomberg.com": 0.92,
    "ft.com": 0.91,
    "wsj.com": 0.90,
    "economictimes.indiatimes.com": 0.87,
    "livemint.com": 0.87,
    "businessstandard.com": 0.86,
    "moneycontrol.com": 0.82,
    "financialexpress.com": 0.82,
    "cnbctv18.com": 0.80,
    "zeebiz.com": 0.75,
    "thehindu.com": 0.85,
    "ndtv.com": 0.78,
    # International financial news
    "cnbc.com": 0.83,
    "marketwatch.com": 0.82,
    "investing.com": 0.78,
    "tradingeconomics.com": 0.80,
    "macrotrends.net": 0.78,
    # Research / academic
    "ssrn.com": 0.88,
    "papers.ssrn.com": 0.88,
    "scholar.google.com": 0.85,
    # Default for unknown domains
    "_default": 0.55,
}


def _score_source(url: str) -> float:
    """Return a credibility score [0–1] for a given URL."""
    try:
        domain = urlparse(url).netloc.lower().lstrip("www.")
    except Exception:
        return _CREDIBILITY_SCORES["_default"]

    # Exact match
    if domain in _CREDIBILITY_SCORES:
        return _CREDIBILITY_SCORES[domain]

    # Partial match (e.g., subdomain.reuters.com)
    for key, score in _CREDIBILITY_SCORES.items():
        if key != "_default" and key in domain:
            return score

    return _CREDIBILITY_SCORES["_default"]


# ---------------------------------------------------------------------------
# Research data containers
# ---------------------------------------------------------------------------


@dataclass
class ResearchFinding:
    """A single piece of information extracted during research."""

    content: str
    source_url: str
    source_name: str
    credibility: float
    query_used: str
    round_number: int
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    relevance_score: float = 0.5
    content_hash: str = field(init=False)

    def __post_init__(self):
        self.content_hash = hashlib.md5(
            self.content[:200].encode(), usedforsecurity=False
        ).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "source_url": self.source_url,
            "source_name": self.source_name,
            "credibility": round(self.credibility, 3),
            "relevance_score": round(self.relevance_score, 3),
            "query_used": self.query_used,
            "round": self.round_number,
            "timestamp": self.timestamp,
        }


@dataclass
class ResearchPlan:
    """The initial research plan generated by the LLM."""

    main_query: str
    sub_questions: List[str]
    search_queries: List[str]
    domain_focus: str
    estimated_rounds: int


# ---------------------------------------------------------------------------
# Main agent
# ---------------------------------------------------------------------------


class WebResearchAgent(BaseSwarmAgent):
    """
    Autonomous web research agent with multi-step reasoning and KB learning.

    Research loop per round:
      1. LLM plans next batch of search queries based on knowledge gaps
      2. Execute searches in parallel
      3. LLM evaluates results: extract key facts, identify gaps
      4. Decide: continue research or synthesise
      5. If new facts: store to KB

    After all rounds:
      • Synthesise all findings into a structured report
      • Persist high-quality findings to Qdrant KB
      • Return full research report with citations
    """

    AGENT_TYPE = AGENT_TYPE
    DEFAULT_TIMEOUT_S = 180.0  # deep research takes time — increased from 120s

    MAX_ROUNDS = 5
    MIN_ROUNDS = 1
    MAX_PARALLEL_SEARCHES = 2  # run 2 searches per round
    MAX_FINDINGS = 30
    MIN_CREDIBILITY_TO_STORE = 0.70  # only store trustworthy sources in KB

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._findings: List[ResearchFinding] = []
        self._searched_queries: set = set()
        self._sources: Dict[str, Dict[str, Any]] = {}  # url → metadata

    # ────────────────────────────────────────────────────────────────────────
    # execute()
    # ────────────────────────────────────────────────────────────────────────

    async def execute(self, message: SwarmMessage) -> AgentResult:
        payload = message.payload
        query: str = str(payload.get("query", "")).strip()
        depth: int = max(
            self.MIN_ROUNDS, min(2, int(payload.get("depth", 2)))  # cap at 2 rounds for speed
        )
        domain_focus: str = str(payload.get("domain_focus", "indian_markets"))
        store_in_kb: bool = bool(payload.get("store_in_kb", True))
        max_results: int = int(payload.get("max_results", 15))

        if not query:
            return self._ok(
                data={"error": "No query provided"},
                summary="Research aborted: empty query.",
                signal="neutral",
                confidence=0.0,
            )

        self._log.info(
            f"🔍 WebResearchAgent: query={query!r} depth={depth} "
            f"domain={domain_focus} store_kb={store_in_kb}"
        )

        # ── Step 1: Plan research ─────────────────────────────────────────
        plan = await self._plan_research(query, depth, domain_focus)
        self._log.info(
            f"📋 Research plan: {len(plan.sub_questions)} sub-questions, "
            f"{len(plan.search_queries)} initial queries"
        )

        # ── Step 2: Execute research rounds ───────────────────────────────
        rounds_done = 0
        pending_queries = list(plan.search_queries)

        for round_num in range(1, depth + 1):
            if not pending_queries:
                self._log.info(f"Round {round_num}: no more queries — stopping early")
                break

            self._log.info(
                f"🌐 Round {round_num}/{depth}: running {len(pending_queries[: self.MAX_PARALLEL_SEARCHES])} searches"
            )

            # Execute this round's searches in parallel
            batch = pending_queries[: self.MAX_PARALLEL_SEARCHES]
            pending_queries = pending_queries[self.MAX_PARALLEL_SEARCHES :]

            new_findings = await self._execute_search_batch(
                batch, round_num, domain_focus
            )
            self._findings.extend(new_findings)
            rounds_done += 1

            # Check if we have enough findings
            if len(self._findings) >= max_results:
                self._log.info(f"Reached max_results={max_results} — stopping research")
                break

            # Generate follow-up queries based on gaps (if more rounds remain)
            if round_num < depth:
                follow_ups = await self._generate_followup_queries(
                    query, self._findings, domain_focus
                )
                pending_queries = follow_ups + pending_queries
                self._log.info(
                    f"🔄 Generated {len(follow_ups)} follow-up queries for round {round_num + 1}"
                )

        # ── Step 3: Deduplicate findings ───────────────────────────────────
        self._findings = self._deduplicate(self._findings)
        self._log.info(f"✂️  After dedup: {len(self._findings)} unique findings")

        # ── Step 4: Synthesise into structured report ─────────────────────
        synthesis = await self._synthesise(query, plan, self._findings, domain_focus)

        # ── Step 5: Store high-quality findings in KB ──────────────────────
        kb_stored = 0
        if store_in_kb and self._findings:
            kb_stored = await self._store_in_knowledge_base(
                query, synthesis, self._findings
            )
            self._log.info(f"📚 Stored {kb_stored} findings in knowledge base")

        # ── Step 6: Build output ───────────────────────────────────────────
        sources_list = self._build_sources_list()
        follow_up_questions = synthesis.get("follow_up_questions", [])
        report_md = synthesis.get("report_md", "")
        confidence = self._compute_confidence()

        return self._ok(
            data={
                "query": query,
                "findings": [f.to_dict() for f in self._findings],
                "sources": sources_list,
                "synthesis": synthesis.get("key_facts", []),
                "report_md": report_md,
                "follow_up_questions": follow_up_questions,
                "knowledge_graph": synthesis.get("knowledge_graph", {}),
                "stored_in_kb": kb_stored,
                "research_rounds": rounds_done,
                "total_findings": len(self._findings),
                "domain_focus": domain_focus,
                "plan": {
                    "sub_questions": plan.sub_questions,
                    "initial_queries": plan.search_queries,
                },
            },
            # Pass full report_md as the summary so downstream agents (ReportAgent)
            # get the complete research context, not just 500 chars
            summary=report_md[:3000]
            if report_md
            else f"Researched {query!r} across {rounds_done} rounds with {len(self._findings)} findings.",
            signal="neutral",
            confidence=confidence,
            metadata={
                "agent": AGENT_TYPE,
                "rounds": rounds_done,
                "findings": len(self._findings),
                "kb_stored": kb_stored,
            },
        )

    # ────────────────────────────────────────────────────────────────────────
    # Step 1: Plan research
    # ────────────────────────────────────────────────────────────────────────

    async def _plan_research(
        self, query: str, depth: int, domain_focus: str
    ) -> ResearchPlan:
        """
        Use LLM to decompose the query into sub-questions and
        generate an initial set of targeted search queries.
        """
        model = self.tools.get_model("fast")

        domain_context = {
            "indian_markets": (
                "Focus on Indian stock markets (NSE, BSE), Indian companies, "
                "SEBI regulations, RBI policy, FII/DII flows, Nifty/Sensex."
            ),
            "global": (
                "Focus on global financial markets: US, Europe, Asia. "
                "S&P 500, Fed policy, global macro, commodities, forex."
            ),
            "crypto": (
                "Focus on cryptocurrency markets: Bitcoin, Ethereum, altcoins, "
                "DeFi, on-chain metrics, regulatory developments."
            ),
            "general": "Broad financial and economic research.",
        }.get(domain_focus, "Financial markets research.")

        prompt = f"""You are an expert financial research strategist.

Query to research: "{query}"

Domain context: {domain_context}

Your job: Create a research plan. Output ONLY valid JSON (no markdown, no extra text):

{{
  "sub_questions": [
    "3-5 specific sub-questions that together fully answer the main query"
  ],
  "search_queries": [
    "6-10 specific Google-style search queries to find answers",
    "Include: current data queries, news queries, analysis queries",
    "Be specific: use company names, dates, numbers when relevant",
    "Mix: NSE/BSE data + news + analyst reports + regulatory filings"
  ],
  "research_strategy": "One sentence on approach"
}}

For Indian market queries, always include:
- "{query} NSE India 2024 2025"
- "{query} SEBI regulation India"
- "{query} analyst report target price"
"""

        try:
            response = await self.tools.call_openrouter(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1200,  # increased from 600 to prevent JSON truncation
            )
            raw = response.choices[0].message.content.strip()

            # Strip markdown code blocks if present
            raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
            raw = re.sub(r"```\s*$", "", raw, flags=re.MULTILINE)

            data = json.loads(raw)
            sub_questions = data.get("sub_questions", [query])[:5]
            search_queries = data.get("search_queries", [query])[:10]

        except Exception as exc:
            self._log.warning(f"Research planning LLM error: {exc} — using fallback")
            sub_questions = [query]
            search_queries = self._fallback_queries(query, domain_focus)

        return ResearchPlan(
            main_query=query,
            sub_questions=sub_questions,
            search_queries=search_queries,
            domain_focus=domain_focus,
            estimated_rounds=depth,
        )

    def _fallback_queries(self, query: str, domain_focus: str) -> List[str]:
        """Fallback search queries when LLM planning fails."""
        base = [
            f"{query} India stock market",
            f"{query} NSE BSE analysis",
            f"{query} latest news 2025",
            f"{query} analyst report forecast",
            f"{query} SEBI RBI regulation",
        ]
        if "nifty" in query.lower() or "sensex" in query.lower():
            base.append(f"{query} technical analysis today")
            base.append("Nifty 50 market outlook FII DII")
        return base

    # ────────────────────────────────────────────────────────────────────────
    # Step 2: Execute searches
    # ────────────────────────────────────────────────────────────────────────

    async def _execute_search_batch(
        self,
        queries: List[str],
        round_num: int,
        domain_focus: str,
    ) -> List[ResearchFinding]:
        """
        Run multiple searches in parallel, parse results, score relevance.
        """
        # Filter already-searched queries
        new_queries = [q for q in queries if q not in self._searched_queries]
        if not new_queries:
            return []

        for q in new_queries:
            self._searched_queries.add(q)

        # Run searches with stagger to avoid hitting 30 RPM rate limit
        # (Groq compound models: 30 RPM, 250 RPD — each search uses ~1 RPM)
        findings: list = []
        for i, q in enumerate(new_queries):
            if i > 0:
                await asyncio.sleep(1.0)  # 1s stagger between searches
            try:
                result = await self._run_single_search(q, round_num)
                if result:
                    findings.extend(result)
            except Exception as exc:
                self._log.debug(
                    f"Search failed for query #{i} in round {round_num}: {exc}"
                )

        return findings

    async def _run_single_search(
        self, query: str, round_num: int
    ) -> List[ResearchFinding]:
        """
        Execute one web search and parse the response into findings.
        """
        try:
            self._log.debug(f"Searching: {query!r}")
            raw_result = await self.tools.web_search(query)

            if not raw_result or len(raw_result.strip()) < 50:
                return []

            # Parse the search result into findings
            findings = await self._parse_search_result(raw_result, query, round_num)
            return findings

        except Exception as exc:
            self._log.warning(f"Search error for {query!r}: {exc}")
            return []

    async def _parse_search_result(
        self,
        raw_result: str,
        query: str,
        round_num: int,
    ) -> List[ResearchFinding]:
        """
        Use LLM to extract structured facts from raw search output.
        Returns list of ResearchFinding objects.
        """
        model = self.tools.get_model("fast")  # use fast model — parsing doesn't need deep reasoning

        prompt = f"""You are a financial data extraction expert.

Search query: "{query}"
Search result:
{raw_result[:3000]}

Extract ONLY factual, verifiable information. Output ONLY valid JSON array (no extra text):

[
  {{
    "fact": "Specific factual statement with numbers/names/dates",
    "source_name": "Publication or website name",
    "source_url": "URL if mentioned, else empty string",
    "relevance": 0.9
  }}
]

Rules:
- Maximum 5 facts per search
- Only include facts directly answering or relevant to: "{query}"
- Include specific numbers, dates, percentages when available
- Skip vague or generic statements
- relevance score: 1.0=directly answers query, 0.5=related context, 0.3=tangential
- If no relevant facts found, return empty array []
"""

        try:
            response = await self.tools.call_openrouter(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=600,  # parsing facts — 600 is plenty, keep it fast
            )
            raw = response.choices[0].message.content.strip()

            # Strip markdown if present
            raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
            raw = re.sub(r"```\s*$", "", raw, flags=re.MULTILINE)

            facts = json.loads(raw)
            if not isinstance(facts, list):
                facts = []

        except Exception as exc:
            self._log.debug(f"Parse LLM error: {exc} — using raw as single finding")
            # Fallback: use entire result as one finding
            facts = [
                {
                    "fact": raw_result[:500],
                    "source_name": "Web Search",
                    "source_url": "",
                    "relevance": 0.5,
                }
            ]

        findings = []
        for item in facts[:5]:
            if not isinstance(item, dict):
                continue
            content = str(item.get("fact", "")).strip()
            if not content or len(content) < 20:
                continue

            source_url = str(item.get("source_url", "")).strip()
            source_name = str(item.get("source_name", "Web Search")).strip()
            relevance = float(item.get("relevance", 0.5))
            credibility = _score_source(source_url) if source_url else 0.55

            finding = ResearchFinding(
                content=content,
                source_url=source_url,
                source_name=source_name,
                credibility=credibility,
                query_used=query,
                round_number=round_num,
                relevance_score=relevance,
            )
            findings.append(finding)

            # Track source
            if source_url and source_url not in self._sources:
                self._sources[source_url] = {
                    "name": source_name,
                    "url": source_url,
                    "credibility": credibility,
                    "findings_count": 0,
                }
            if source_url in self._sources:
                self._sources[source_url]["findings_count"] += 1

        return findings

    # ────────────────────────────────────────────────────────────────────────
    # Step 2b: Generate follow-up queries based on knowledge gaps
    # ────────────────────────────────────────────────────────────────────────

    async def _generate_followup_queries(
        self,
        main_query: str,
        findings: List[ResearchFinding],
        domain_focus: str,
    ) -> List[str]:
        """
        Ask the LLM: given what we know so far, what should we search next?
        Self-directed research loop — the core of the agentic behavior.
        """
        if not findings:
            return []

        model = self.tools.get_model("fast")

        # Summarise current knowledge
        known_facts = "\n".join(f"- {f.content[:150]}" for f in findings[-10:])

        prompt = f"""You are a financial research agent reviewing your progress.

Original research question: "{main_query}"

What you know so far:
{known_facts}

Identify the TOP 3 most important GAPS in this research.
Generate 3-4 highly specific search queries to fill those gaps.

Output ONLY valid JSON (no extra text):
{{
  "gaps": ["gap 1", "gap 2", "gap 3"],
  "follow_up_queries": [
    "specific search query 1",
    "specific search query 2",
    "specific search query 3"
  ]
}}

Rules:
- Follow-up queries must NOT repeat information already found
- Be specific: include company names, dates, numbers
- Focus on: missing data points, unanswered sub-questions, conflicting information
- For Indian markets: include "India NSE" or specific company ticker when relevant
"""

        try:
            response = await self.tools.call_openrouter(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=400,
            )
            raw = response.choices[0].message.content.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
            raw = re.sub(r"```\s*$", "", raw, flags=re.MULTILINE)

            data = json.loads(raw)
            queries = data.get("follow_up_queries", [])
            # Filter out already searched
            new_queries = [q for q in queries if q not in self._searched_queries][:4]
            return new_queries

        except Exception as exc:
            self._log.debug(f"Follow-up generation error: {exc}")
            return []

    # ────────────────────────────────────────────────────────────────────────
    # Step 3: Deduplicate findings
    # ────────────────────────────────────────────────────────────────────────

    def _deduplicate(self, findings: List[ResearchFinding]) -> List[ResearchFinding]:
        """
        Remove duplicate or near-duplicate findings using content hash
        and simple Jaccard similarity on word sets.
        """
        seen_hashes: set = set()
        unique: List[ResearchFinding] = []

        for finding in sorted(
            findings,
            key=lambda f: f.credibility * f.relevance_score,
            reverse=True,
        ):
            # Hash dedup
            if finding.content_hash in seen_hashes:
                continue

            # Simple near-dedup: check against last 10 uniques
            is_dupe = False
            words_a = set(finding.content.lower().split())
            for existing in unique[-10:]:
                words_b = set(existing.content.lower().split())
                union = words_a | words_b
                if not union:
                    continue
                jaccard = len(words_a & words_b) / len(union)
                if jaccard > 0.70:  # 70% word overlap = near-duplicate
                    is_dupe = True
                    break

            if not is_dupe:
                seen_hashes.add(finding.content_hash)
                unique.append(finding)

        return unique

    # ────────────────────────────────────────────────────────────────────────
    # Step 4: Synthesise findings into structured report
    # ────────────────────────────────────────────────────────────────────────

    async def _synthesise(
        self,
        query: str,
        plan: ResearchPlan,
        findings: List[ResearchFinding],
        domain_focus: str,
    ) -> Dict[str, Any]:
        """
        Use LLM to synthesise all findings into:
          - key_facts: bullet list of confirmed facts
          - report_md: rich markdown research report
          - knowledge_graph: entities and relationships extracted
          - follow_up_questions: open questions for further research
          - directional_signal: market signal if applicable
        """
        if not findings:
            return {
                "key_facts": [],
                "report_md": f"No data found for: {query}",
                "knowledge_graph": {},
                "follow_up_questions": [],
                "directional_signal": "neutral",
            }

        model = self.tools.get_model("reasoning")

        # Build findings text (sorted by credibility × relevance)
        sorted_findings = sorted(
            findings,
            key=lambda f: f.credibility * f.relevance_score,
            reverse=True,
        )[:20]  # top 20 for synthesis

        findings_text = "\n\n".join(
            f"[Source: {f.source_name} | Credibility: {f.credibility:.2f} | Relevance: {f.relevance_score:.2f}]\n{f.content}"
            for f in sorted_findings
        )

        sub_questions_text = "\n".join(
            f"  {i + 1}. {q}" for i, q in enumerate(plan.sub_questions)
        )

        system_prompt = """You are Daddy's AI — India's most advanced financial research analyst.
You have just completed multi-round autonomous web research with verified sources.
Synthesise all findings into a comprehensive, citation-rich research report.

Output style required:
- Lead with the most important insight in the FIRST sentence (include a specific number or date)
- Use inline citations: "According to Economic Times (87% credibility)..."
- Structure: Executive Insight → Key Facts → Latest News & Events → Market Context → Risks → Forward Outlook
- Every claim MUST have a specific number, date, or named source
- Identify contradictions across sources and explain which is more credible and why
- The report_md field MUST be minimum 2000 words — do NOT truncate; give the FULL analysis
- Use ## headers, **bold** for key numbers, and tables where it helps clarity
- End with "Follow-up research areas:" listing 3-5 open questions"""

        user_prompt = f"""Research query: "{query}"

Sub-questions to answer:
{sub_questions_text}

Research findings ({len(sorted_findings)} sources):
{findings_text}

Synthesise this research into a comprehensive report. Output ONLY valid JSON (no extra text):

{{
  "key_facts": [
    "Bullet 1: Most important verified fact with specific numbers and source name",
    "Bullet 2: ...",
    "... up to 10 key facts, each 1-2 sentences with data"
  ],
  "report_md": "Full markdown research report MINIMUM 2000 WORDS. Use ## headers. Bold key numbers. Cite every claim. Cover: latest news, price action context, company-specific developments, sector trends, risks, and what it means for investors. Lead with the single most important finding with a specific number in sentence 1.",
  "knowledge_graph": {{
    "entities": ["Company A", "Person B", "Event C"],
    "relationships": [["Company A", "acquired", "Company B"]]
  }},
  "follow_up_questions": [
    "Specific follow-up question 1 with a number or name",
    "Specific follow-up question 2",
    "Specific follow-up question 3"
  ],
  "directional_signal": "bullish | bearish | neutral",
  "signal_reasoning": "One precise sentence with numbers explaining the directional signal",
  "confidence": 0.75
}}"""

        try:
            response = await self.tools.call_openrouter(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=3500,  # 3500 is plenty for synthesising 5-10 findings quickly
            )
            raw = response.choices[0].message.content.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
            raw = re.sub(r"```\s*$", "", raw, flags=re.MULTILINE)

            result = json.loads(raw)
            return result

        except Exception as exc:
            self._log.warning(f"Synthesis LLM error: {exc} — building simple report")
            return self._fallback_synthesis(query, findings)

    def _fallback_synthesis(
        self, query: str, findings: List[ResearchFinding]
    ) -> Dict[str, Any]:
        """Simple synthesis when LLM fails."""
        key_facts = [f.content[:200] for f in findings[:8]]
        report_lines = [
            f"## 🔍 Research Report: {query}",
            "",
            f"**Findings from {len(findings)} sources:**",
            "",
        ]
        for f in findings[:8]:
            report_lines.append(f"- {f.content[:200]}")

        report_lines.extend(
            [
                "",
                "⚠️ *Research for informational purposes only.*",
            ]
        )

        return {
            "key_facts": key_facts,
            "report_md": "\n".join(report_lines),
            "knowledge_graph": {"entities": [], "relationships": []},
            "follow_up_questions": [],
            "directional_signal": "neutral",
            "signal_reasoning": "Insufficient data for directional signal.",
            "confidence": 0.4,
        }

    # ────────────────────────────────────────────────────────────────────────
    # Step 5: Store findings in KB
    # ────────────────────────────────────────────────────────────────────────

    async def _store_in_knowledge_base(
        self,
        query: str,
        synthesis: Dict[str, Any],
        findings: List[ResearchFinding],
    ) -> int:
        """
        Persist high-quality research findings to the Qdrant KB.

        Stores:
          1. The synthesised report (as a single rich document)
          2. Individual high-credibility findings (credibility >= threshold)

        Returns: number of items stored
        """
        stored = 0
        try:
            from ...rag import get_kb_rag

            kb = get_kb_rag()
            if not getattr(kb, "_ready", False):
                self._log.debug("KB not ready (Qdrant offline) — skipping KB storage")
                return 0

            # Store the synthesised report
            report_md = synthesis.get("report_md", "")
            if report_md and len(report_md) > 100:
                doc_text = (
                    f"RESEARCH REPORT — Query: {query}\n\n"
                    f"{report_md}\n\n"
                    f"Key Facts:\n"
                    + "\n".join(f"• {f}" for f in synthesis.get("key_facts", []))
                )
                await self._upsert_to_kb(
                    kb,
                    doc_text,
                    {
                        "type": "research_report",
                        "query": query,
                        "timestamp": datetime.utcnow().isoformat(),
                        "confidence": synthesis.get("confidence", 0.5),
                        "signal": synthesis.get("directional_signal", "neutral"),
                    },
                )
                stored += 1

            # Store individual high-credibility findings
            for finding in findings:
                if finding.credibility >= self.MIN_CREDIBILITY_TO_STORE:
                    doc_text = (
                        f"SOURCE: {finding.source_name}\n"
                        f"QUERY: {finding.query_used}\n"
                        f"FACT: {finding.content}"
                    )
                    await self._upsert_to_kb(
                        kb,
                        doc_text,
                        {
                            "type": "research_finding",
                            "source_url": finding.source_url,
                            "source_name": finding.source_name,
                            "credibility": finding.credibility,
                            "relevance": finding.relevance_score,
                            "query": finding.query_used,
                            "timestamp": finding.timestamp,
                        },
                    )
                    stored += 1

        except ImportError:
            self._log.debug("RAG module not available — skipping KB storage")
        except Exception as exc:
            self._log.warning(f"KB storage error: {exc}")

        return stored

    async def _upsert_to_kb(
        self,
        kb: Any,
        text: str,
        metadata: Dict[str, Any],
    ) -> None:
        """
        Upsert a single document to the knowledge base.
        Uses the KB's add_document method if available, otherwise
        falls back to the ingest pipeline.
        """
        try:
            # Try direct add method first
            if hasattr(kb, "add_document"):
                await kb.add_document(text=text, metadata=metadata)
            elif hasattr(kb, "upsert"):
                await kb.upsert(text=text, metadata=metadata)
            else:
                # Fallback: use the ingest pipeline
                from ...rag.ingest_kb import ingest_text

                await ingest_text(text=text, metadata=metadata)
        except Exception as exc:
            self._log.debug(f"KB upsert failed (non-critical): {exc}")

    # ────────────────────────────────────────────────────────────────────────
    # Helpers
    # ────────────────────────────────────────────────────────────────────────

    def _build_sources_list(self) -> List[Dict[str, Any]]:
        """Build a deduplicated, sorted list of all sources used."""
        sources = sorted(
            self._sources.values(),
            key=lambda s: s["credibility"],
            reverse=True,
        )
        return [
            {
                "name": s["name"],
                "url": s["url"],
                "credibility": round(s["credibility"], 3),
                "findings_count": s["findings_count"],
            }
            for s in sources
        ]

    def _compute_confidence(self) -> float:
        """
        Compute overall research confidence based on:
          - Number of findings
          - Average credibility × relevance
          - Source diversity
        """
        if not self._findings:
            return 0.1

        avg_quality = sum(
            f.credibility * f.relevance_score for f in self._findings
        ) / len(self._findings)

        # Bonus for more findings (up to 15)
        volume_bonus = min(1.0, len(self._findings) / 15) * 0.2

        # Bonus for source diversity
        unique_sources = len(self._sources)
        diversity_bonus = min(1.0, unique_sources / 8) * 0.1

        confidence = avg_quality * 0.7 + volume_bonus + diversity_bonus
        return round(min(1.0, max(0.0, confidence)), 3)

    async def teardown(self) -> None:
        """Clean up any state on disposal."""
        self._findings.clear()
        self._searched_queries.clear()
        self._sources.clear()
