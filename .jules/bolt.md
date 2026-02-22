## 2025-01-30 - NSE Scraper Optimization
**Learning:** `httpx.AsyncClient` must be reused to benefit from connection pooling and session persistence. Creating a new client for every request is a significant bottleneck. Additionally, `Accept-Encoding: br` causes decoding errors in `httpx` if the `brotli` library is not installed, even if `httpx` is used correctly.
**Action:** Always use a persistent `httpx.AsyncClient` (preferably as a singleton or in a lifespan manager) and ensure `Accept-Encoding` matches installed decompression libraries.

## 2025-01-30 - Parallel Tool Execution in Agents
**Learning:** Agents often call multiple tools in a single turn (e.g., fetching a quote and searching the web). Sequential execution of these tools is a major latency bottleneck. Using `asyncio.gather` to parallelize independent tool calls can improve response times by 50% or more.
**Action:** Use `asyncio.gather` for multiple autonomous tool calls in agents. Ensure the assistant message is appended to history once, followed by all tool results in the correct format.

## 2025-01-31 - Groq Client Connection Pooling
**Learning:** `AsyncGroq` (and the underlying `httpx.AsyncClient`) should be reused to benefit from connection pooling (reusing TCP/TLS sessions). Creating a new client for every completion or web search is a significant latency bottleneck. The Groq SDK uses `.close()` as the asynchronous method for cleanup, not `.aclose()`.
**Action:** Cache and reuse `AsyncGroq` instances in the `GroqKeyRotator`. Ensure graceful shutdown by calling `.close()` in the application lifespan manager.
