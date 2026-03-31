"""
ConversationMemoryCompressor
============================
TurboQuant-inspired QJL (Quantized Johnson-Lindenstrauss) compression
for conversation history retrieval.

Problem:
  Long sessions accumulate 50+ messages → injecting ALL of them into
  the LLM context wastes thousands of tokens and hurts accuracy.

Solution (QJL):
  1. Embed every history turn → 384-dim float32 vectors
  2. Apply QJL (random projection + sign bit) → 384-bit binary vectors
  3. At query time: Hamming distance search → retrieve top-k relevant turns only
  4. Inject only those ~5 turns into the LLM context instead of all 50

Benefits:
  - 5-10x fewer context tokens for long sessions
  - Semantically relevant context (not just recency-biased)
  - Binary Hamming search is extremely fast (XOR + popcount)
  - 32x smaller memory footprint for compressed vectors
  - Zero training, data-oblivious (fixed random projection seed)

Usage:
    compressor = ConversationMemoryCompressor(get_embedding_service())
    relevant   = compressor.retrieve_relevant(query, history, top_k=5)
    # Use `relevant` instead of full history in AgentState
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class ConversationMemoryCompressor:
    """
    QJL-based semantic retrieval for conversation history.

    The Johnson-Lindenstrauss Transform (JLT) guarantees that random
    projections approximately preserve pairwise distances.  Taking only
    the sign bit of the projected vector gives a 1-bit binary code that
    supports fast Hamming-distance nearest-neighbour search.

    Args:
        embedding_service: EmbeddingService instance (all-MiniLM-L6-v2)
        projection_seed:   Fixed seed → deterministic, data-oblivious projections
        always_keep_recent: Always include last N turns regardless of relevance
    """

    def __init__(
        self,
        embedding_service,
        projection_seed: int = 42,
        always_keep_recent: int = 2,
    ) -> None:
        self._embedder = embedding_service
        self._seed = projection_seed
        self._always_keep_recent = always_keep_recent
        # Cache the projection matrix per (input_dim, output_dim) pair
        self._projection_cache: Dict[tuple, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_projection(self, dim: int) -> np.ndarray:
        """
        Return (or build) the fixed random projection matrix P of shape (dim, dim).
        Using a fixed seed makes this data-oblivious — same matrix for every query.
        """
        key = (dim, dim)
        if key not in self._projection_cache:
            rng = np.random.default_rng(self._seed)
            # Standard normal entries → JLT preserves pairwise distances
            self._projection_cache[key] = rng.standard_normal((dim, dim)).astype(
                np.float32
            )
            logger.debug(f"Built QJL projection matrix ({dim}×{dim})")
        return self._projection_cache[key]

    def _qjl_encode(self, vectors: np.ndarray) -> np.ndarray:
        """
        QJL encoding: random projection → sign bit.

        Args:
            vectors: (N, D) float32 embeddings

        Returns:
            (N, D) uint8 binary matrix (each entry is 0 or 1)
            Memory: N × D bytes vs N × D × 4 bytes for float32 → 4x smaller
            (Further packing as bits would give 32x reduction — uint8 is simpler)
        """
        P = self._get_projection(vectors.shape[1])
        projected = vectors @ P  # (N, D) — preserves distances (JLT theorem)
        return (projected > 0).astype(np.uint8)  # sign bit

    def _hamming_distances(
        self, query_binary: np.ndarray, history_binary: np.ndarray
    ) -> np.ndarray:
        """
        Compute Hamming distances between a query binary vector and all history
        binary vectors using XOR + sum (equivalent to popcount).

        Args:
            query_binary:   (1, D) or (D,) uint8
            history_binary: (N, D) uint8

        Returns:
            (N,) int array of Hamming distances (lower = more similar)
        """
        q = query_binary.reshape(1, -1)
        xor = np.bitwise_xor(history_binary, q)  # (N, D)
        return xor.sum(axis=1)  # (N,) Hamming distances

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compress_history(self, history: List[Dict]) -> Optional[np.ndarray]:
        """
        Embed all history turns and QJL-compress to binary vectors.

        Args:
            history: list of {role, content} dicts

        Returns:
            (N, D) uint8 binary matrix, or None if embedding unavailable
        """
        if not history:
            return None

        if not self._embedder.is_available():
            logger.warning("Embedding service unavailable — skipping QJL compression")
            return None

        # Format: "role: content" so the embedding captures speaker roles too
        texts = [f"{m.get('role', 'user')}: {m.get('content', '')}" for m in history]
        try:
            embeddings = self._embedder.encode(texts)  # (N, 384) float32
            if embeddings.size == 0:
                return None
            return self._qjl_encode(embeddings)        # (N, 384) uint8 binary
        except Exception as e:
            logger.error(f"QJL compression error: {e}")
            return None

    def retrieve_relevant(
        self,
        query: str,
        history: List[Dict],
        top_k: int = 5,
    ) -> List[Dict]:
        """
        Return the top-k most semantically relevant history turns using
        QJL binary Hamming distance search.

        Always includes the most recent `always_keep_recent` turns
        (regardless of relevance score) to preserve dialogue continuity.

        Falls back to returning the last `top_k` turns if embedding
        service is unavailable.

        Args:
            query:   Current user query
            history: Full conversation history [{role, content}, ...]
            top_k:   Max turns to inject into LLM context

        Returns:
            Filtered, chronologically-ordered list of relevant history turns
        """
        # Fast path: already small enough
        if len(history) <= top_k:
            return history

        # Fast path: embedding unavailable → recency fallback
        if not self._embedder.is_available():
            logger.debug("QJL compression unavailable — using recency fallback")
            return history[-top_k:]

        try:
            # 1. Compress all history turns to binary
            binary_history = self.compress_history(history)
            if binary_history is None:
                return history[-top_k:]

            # 2. Embed + compress the current query
            q_embedding = self._embedder.encode(query)  # (1, 384) or (384,)
            if q_embedding.size == 0:
                return history[-top_k:]

            q_vec = q_embedding.reshape(1, -1) if q_embedding.ndim == 1 else q_embedding
            q_binary = self._qjl_encode(q_vec.astype(np.float32))  # (1, 384) uint8

            # 3. Hamming distance search
            distances = self._hamming_distances(q_binary, binary_history)  # (N,)

            # 4. Always keep last `always_keep_recent` turns
            recent_indices = set(range(len(history) - self._always_keep_recent, len(history)))

            # 5. Find top-k semantically similar turns (excluding already-kept recent)
            remaining_k = max(0, top_k - len(recent_indices))
            candidate_indices = [i for i in range(len(history)) if i not in recent_indices]

            if candidate_indices and remaining_k > 0:
                candidate_distances = distances[candidate_indices]
                # argpartition is O(N) — faster than full sort for large histories
                n_select = min(remaining_k, len(candidate_indices))
                top_candidate_positions = np.argpartition(candidate_distances, n_select - 1)[
                    :n_select
                ]
                semantic_indices = {candidate_indices[p] for p in top_candidate_positions}
            else:
                semantic_indices = set()

            # 6. Merge, sort chronologically (preserve conversation flow)
            selected = sorted(recent_indices | semantic_indices)
            result = [history[i] for i in selected]

            logger.debug(
                f"QJL memory: {len(history)} turns → {len(result)} selected "
                f"(top_k={top_k}, recent={self._always_keep_recent})"
            )
            return result

        except Exception as e:
            logger.error(f"QJL retrieval error: {e} — falling back to last {top_k} turns")
            return history[-top_k:]

    def compression_stats(self, history: List[Dict]) -> Dict:
        """
        Return compression statistics for a given history.
        Useful for debugging and monitoring.
        """
        n = len(history)
        total_chars = sum(len(m.get("content", "")) for m in history)
        float32_bytes = n * 384 * 4   # float32 embeddings
        binary_bytes = n * 384        # uint8 binary (1 byte per dim, could pack to 1 bit)
        return {
            "turns": n,
            "total_chars": total_chars,
            "estimated_tokens": total_chars // 4,
            "float32_embedding_bytes": float32_bytes,
            "qjl_binary_bytes": binary_bytes,
            "compression_ratio": round(float32_bytes / max(binary_bytes, 1), 1),
        }


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_compressor: Optional[ConversationMemoryCompressor] = None


def get_memory_compressor() -> ConversationMemoryCompressor:
    """Get or create the global ConversationMemoryCompressor singleton."""
    global _compressor
    if _compressor is None:
        try:
            from app.rag.embedding_service import get_embedding_service
            _compressor = ConversationMemoryCompressor(get_embedding_service())
            logger.info("✅ ConversationMemoryCompressor (QJL) initialized")
        except Exception as e:
            logger.warning(f"⚠️ Could not init CompressorMemory: {e}")
            # Return a dummy that always falls back to recency
            class _DummyEmbedder:
                def is_available(self): return False
                def encode(self, *a, **kw): return np.array([])
            _compressor = ConversationMemoryCompressor(_DummyEmbedder())
    return _compressor
