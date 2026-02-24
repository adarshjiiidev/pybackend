"""
Qdrant Cloud-based Knowledge Base RAG.
Primary: semantic cosine search on Qdrant Cloud.
Fallback: keyword-based KnowledgeBaseRAG (knowledge_index.py).
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class QdrantKBRAG:
    """
    Semantic KB search using Qdrant Cloud + sentence-transformers.
    - Primary  : Qdrant Cloud cosine similarity search
    - Fallback : keyword-based KnowledgeBaseRAG (always available)
    - Safety   : constraints.txt always included in results
    """

    def __init__(self):
        self._client = None
        self._embedder = None
        self._ready = False
        self._fallback = None
        self._collection = "daddys_kb"
        self._initialize()

    def _initialize(self):
        """Connect to Qdrant Cloud and load embedder. Gracefully falls back on any error."""
        try:
            from qdrant_client import QdrantClient
            from sentence_transformers import SentenceTransformer
            from app.config.settings import settings

            qdrant_url = settings.qdrant_url
            qdrant_api_key = settings.qdrant_api_key
            self._collection = settings.qdrant_collection

            if not qdrant_url or not qdrant_api_key:
                logger.warning("⚠️  QDRANT_URL or QDRANT_API_KEY not set — using keyword fallback")
                self._load_fallback()
                return

            # Check collection exists
            self._client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
            collections = [c.name for c in self._client.get_collections().collections]
            if self._collection not in collections:
                logger.warning(
                    f"⚠️  Qdrant collection '{self._collection}' not found. "
                    "Run: python -m app.rag.ingest_kb — using keyword fallback for now"
                )
                self._load_fallback()
                return

            info = self._client.get_collection(self._collection)
            logger.info(f"✅ Qdrant Cloud KB ready: {info.points_count} chunks in '{self._collection}'")

            logger.info("🔄 Loading sentence-transformers embedder (all-MiniLM-L6-v2)...")
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("✅ Embedder loaded")

            self._ready = True

            # Also load keyword fallback silently (used if Qdrant search throws at runtime)
            self._load_fallback(silent=True)

        except Exception as e:
            logger.error(f"❌ Qdrant Cloud init failed: {e} — using keyword fallback")
            self._load_fallback()

    def _load_fallback(self, silent: bool = False):
        """Load the original keyword-based RAG as fallback."""
        try:
            from .knowledge_base import KnowledgeBaseRAG
            self._fallback = KnowledgeBaseRAG()
            if not silent:
                logger.info("✅ Keyword fallback RAG loaded")
        except Exception as e:
            logger.error(f"Keyword fallback RAG also failed: {e}")

    def _embed(self, text: str) -> List[float]:
        vec = self._embedder.encode([text], show_progress_bar=False)
        return vec[0].tolist()

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Search: Qdrant Cloud first → keyword fallback if unavailable.
        Always includes constraints.txt as the first result.
        """
        if self._ready and self._client and self._embedder:
            try:
                return self._qdrant_search(query, top_k)
            except Exception as e:
                logger.error(f"Qdrant runtime search error: {e} — falling back to keywords")

        # Fallback
        if self._fallback:
            logger.debug("KB: using keyword fallback search")
            return self._fallback.search(query, top_k)

        logger.error("KB: both Qdrant and keyword fallback unavailable")
        return []

    def _qdrant_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        query_vec = self._embed(query)

        hits = self._client.search(
            collection_name=self._collection,
            query_vector=query_vec,
            limit=top_k * 5,          # Wider net for deduplication
            score_threshold=0.20,     # Min relevance (0=random, 1=identical)
        )

        seen_files: Dict[str, Dict] = {}
        constraints_result = None

        for hit in hits:
            p = hit.payload or {}
            fname = p.get("filename", "unknown")
            score = hit.score
            doc = {
                "filename": fname,
                "title": p.get("title", fname),
                "content": p.get("content", ""),
                "score": score,
            }

            if p.get("is_constraints"):
                if constraints_result is None or score > constraints_result["score"]:
                    constraints_result = doc
                continue

            # Keep best-scoring chunk per file
            if fname not in seen_files or score > seen_files[fname]["score"]:
                seen_files[fname] = doc

        # Build ordered results: constraints first, then top-k files
        if constraints_result is None:
            constraints_result = self._read_constraints_directly()

        results = []
        if constraints_result:
            results.append(constraints_result)

        sorted_docs = sorted(seen_files.values(), key=lambda x: x["score"], reverse=True)
        results.extend(sorted_docs[: max(top_k - 1, 1)])

        logger.info(
            f"🔍 Qdrant KB: '{query[:50]}' → {len(results)} results "
            f"(scores: {[round(r['score'], 2) for r in results]})"
        )
        return results

    def _read_constraints_directly(self) -> Optional[Dict]:
        """Safety net: read constraints.txt directly from disk."""
        try:
            p = Path(__file__).parent.parent.parent / "txt" / "constraints.txt"
            if p.exists():
                return {
                    "filename": "constraints.txt",
                    "title": "Constraints",
                    "content": p.read_text(encoding="utf-8")[:800],
                    "score": 1.0,
                }
        except Exception:
            pass
        return None

    def get_relevant_context(self, query: str, max_chars: int = 3000) -> str:
        """Formatted context string for LLM prompt injection."""
        results = self.search(query, top_k=3)
        if not results:
            return ""

        parts = []
        total = 0
        for r in results:
            header = f"\n## {r['title']} ({r['filename']})\n"
            available = max_chars - total - len(header)
            if available <= 0:
                break
            content = r["content"][:available]
            parts.append(header + content)
            total += len(header) + len(content)

        return "\n---\n**Knowledge Base Context:**\n" + "\n".join(parts) if parts else ""


# --- Global singleton ---
_qdrant_rag: Optional[QdrantKBRAG] = None


def get_qdrant_rag() -> QdrantKBRAG:
    """Get or create global QdrantKBRAG singleton."""
    global _qdrant_rag
    if _qdrant_rag is None:
        _qdrant_rag = QdrantKBRAG()
    return _qdrant_rag
