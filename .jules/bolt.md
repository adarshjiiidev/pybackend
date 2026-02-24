## 2025-01-30 - NSE Scraper Optimization
**Learning:** `httpx.AsyncClient` must be reused to benefit from connection pooling and session persistence. Creating a new client for every request is a significant bottleneck. Additionally, `Accept-Encoding: br` causes decoding errors in `httpx` if the `brotli` library is not installed, even if `httpx` is used correctly.
**Action:** Always use a persistent `httpx.AsyncClient` (preferably as a singleton or in a lifespan manager) and ensure `Accept-Encoding` matches installed decompression libraries.

## 2025-01-30 - Parallel Tool Execution
**Learning:** Sequential tool execution in agents is a major bottleneck. Parallelizing tool calls using `asyncio.gather` can reduce response latency from O(n) to O(1) relative to the number of tools called. When using persistent clients with concurrent calls, initialization must be thread-safe (using `asyncio.Lock`) to avoid redundant client creation.
**Action:** Always execute multiple independent tool calls in parallel and protect shared resource initialization with locks.

## 2025-01-30 - Auth Flow Caching & Backgrounding
**Learning:** Authenticated request latency is often dominated by redundant database lookups for User and Token Blacklist status. Caching these for a short TTL (e.g., 5 mins) in-memory can make request authorization near-instant. Additionally, auxiliary tasks like updating `last_login` or sending welcome emails should always be backgrounded to minimize response time for core actions like login and registration.
**Action:** Implement short-lived in-memory caching for auth dependencies and use `BackgroundTasks` for non-critical state updates.
