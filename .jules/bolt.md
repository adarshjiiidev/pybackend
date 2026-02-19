## 2025-01-30 - NSE Scraper Optimization
**Learning:** `httpx.AsyncClient` must be reused to benefit from connection pooling and session persistence. Creating a new client for every request is a significant bottleneck. Additionally, `Accept-Encoding: br` causes decoding errors in `httpx` if the `brotli` library is not installed, even if `httpx` is used correctly.
**Action:** Always use a persistent `httpx.AsyncClient` (preferably as a singleton or in a lifespan manager) and ensure `Accept-Encoding` matches installed decompression libraries.
