## 2025-01-30 - NSE Scraper Optimization
**Learning:** `httpx.AsyncClient` must be reused to benefit from connection pooling and session persistence. Creating a new client for every request is a significant bottleneck. Additionally, `Accept-Encoding: br` causes decoding errors in `httpx` if the `brotli` library is not installed, even if `httpx` is used correctly.
**Action:** Always use a persistent `httpx.AsyncClient` (preferably as a singleton or in a lifespan manager) and ensure `Accept-Encoding` matches installed decompression libraries.

## 2025-01-30 - Parallel Tool Execution in Agents
**Learning:** Agents often call multiple tools in a single turn (e.g., fetching a quote and searching the web). Sequential execution of these tools is a major latency bottleneck. Using `asyncio.gather` to parallelize independent tool calls can improve response times by 50% or more.
**Action:** Use `asyncio.gather` for multiple autonomous tool calls in agents. Ensure the assistant message is appended to history once, followed by all tool results in the correct format.

## 2025-01-30 - Shared Groq Client Caching
**Learning:** Even with an agent-based architecture where agents are initialized once, tools and other utility functions often create transient `AsyncGroq` clients. Centralizing client creation in the `GroqKeyRotator` with a per-API-key cache ensures that all parts of the application share the same connection pools, reducing handshake latency by 50-100ms per request.
**Action:** Always use `get_groq_client()` from the rotator instead of manual instantiation. Ensure `aclose_all()` is called during application shutdown to prevent resource leaks.

## 2025-01-30 - Efficient History Retrieval & Graph Reuse
**Learning:** Fetching full conversation history to extract the last N messages is a major bottleneck (O(N) database I/O). Using MongoDB's `.sort("-created_at").limit(N)` optimizes this to O(1) relative to conversation length. Additionally, reusing a pre-compiled LangGraph instance prevents redundant graph construction overhead on every request.
**Action:** Use limited, sorted queries for history retrieval. Always reuse pre-compiled global agent graphs.
