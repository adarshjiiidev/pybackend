"""
Lightweight Embedding Service for Knowledge Base Vector Search
Uses sentence-transformers (all-MiniLM-L6-v2) for semantic similarity.
"""

import logging
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import pickle
import hashlib

logger = logging.getLogger(__name__)

# Try to import sentence-transformers, fall back gracefully
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    logger.warning("sentence-transformers not installed. Vector search disabled.")
    EMBEDDINGS_AVAILABLE = False


class EmbeddingService:
    """
    Fast, lightweight embedding service using sentence-transformers.
    Provides semantic search capabilities for knowledge base.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_dir: str = ".cache/embeddings"):
        """
        Initialize embedding service with specified model.
        
        Args:
            model_name: HuggingFace model name (default: all-MiniLM-L6-v2, 384 dims, 40MB)
            cache_dir: Directory to cache embeddings
        """
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.embedding_cache = {}
        
        if EMBEDDINGS_AVAILABLE:
            try:
                self.model = SentenceTransformer(model_name)
                logger.info(f"✅ Loaded embedding model: {model_name}")
                self._load_cache()
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                self.model = None
        else:
            logger.warning("⚠️ Embeddings not available. Install: pip install sentence-transformers")
    
    def is_available(self) -> bool:
        """Check if embedding service is ready."""
        return self.model is not None
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key from text."""
        return hashlib.md5(text.encode()).hexdigest()
    
    def _load_cache(self):
        """Load cached embeddings from disk."""
        cache_file = self.cache_dir / f"{self.model_name.replace('/', '_')}_cache.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    self.embedding_cache = pickle.load(f)
                logger.info(f"📦 Loaded {len(self.embedding_cache)} cached embeddings")
            except Exception as e:
                logger.warning(f"Failed to load embedding cache: {e}")
                self.embedding_cache = {}
    
    def _save_cache(self):
        """Save embeddings cache to disk."""
        cache_file = self.cache_dir / f"{self.model_name.replace('/', '_')}_cache.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(self.embedding_cache, f)
            logger.debug(f"💾 Saved {len(self.embedding_cache)} embeddings to cache")
        except Exception as e:
            logger.warning(f"Failed to save embedding cache: {e}")
    
    def encode(self, texts: List[str] | str, use_cache: bool = True) -> np.ndarray:
        """
        Generate embeddings for text(s).
        
        Args:
            texts: Single text or list of texts
            use_cache: Use cached embeddings if available
            
        Returns:
            numpy array of embeddings (384-dimensional vectors)
        """
        if not self.is_available():
            logger.warning("Embedding service not available, returning empty array")
            return np.array([])
        
        # Handle single text
        if isinstance(texts, str):
            texts = [texts]
        
        # Check cache
        if use_cache:
            uncached_texts = []
            uncached_indices = []
            embeddings = [None] * len(texts)
            
            for i, text in enumerate(texts):
                cache_key = self._get_cache_key(text)
                if cache_key in self.embedding_cache:
                    embeddings[i] = self.embedding_cache[cache_key]
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
            
            # Generate embeddings for uncached texts
            if uncached_texts:
                new_embeddings = self.model.encode(uncached_texts, convert_to_numpy=True)
                
                for idx, text, embedding in zip(uncached_indices, uncached_texts, new_embeddings):
                    cache_key = self._get_cache_key(text)
                    self.embedding_cache[cache_key] = embedding
                    embeddings[idx] = embedding
                
                # Save cache periodically
                if len(uncached_texts) > 0:
                    self._save_cache()
            
            return np.array(embeddings)
        else:
            # Generate without caching
            return self.model.encode(texts, convert_to_numpy=True)
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.
        
        Returns:
            Similarity score between 0 and 1 (1 = identical)
        """
        if vec1.size == 0 or vec2.size == 0:
            return 0.0
        
        dot_product = np.dot(vec1, vec2)
        norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        
        if norm_product == 0:
            return 0.0
        
        return float(dot_product / norm_product)
    
    def find_most_similar(
        self,
        query: str,
        documents: List[Dict[str, any]],
        top_k: int = 3,
        doc_text_key: str = "content"
    ) -> List[Tuple[Dict, float]]:
        """
        Find most similar documents to query using cosine similarity.
        
        Args:
            query: Search query
            documents: List of documents (must have text content)
            top_k: Number of results to return
            doc_text_key: Key in document dict containing text
            
        Returns:
            List of (document, similarity_score) tuples, sorted by similarity
        """
        if not self.is_available() or not documents:
            return []
        
        # Generate query embedding
        query_embedding = self.encode(query)[0]
        
        # Generate document embeddings
        doc_texts = [doc[doc_text_key] for doc in documents]
        doc_embeddings = self.encode(doc_texts)
        
        # Compute similarities
        similarities = []
        for doc, doc_embedding in zip(documents, doc_embeddings):
            similarity = self.cosine_similarity(query_embedding, doc_embedding)
            similarities.append((doc, similarity))
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def batch_encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode large batches of text efficiently.
        
        Args:
            texts: List of texts to encode
            batch_size: Batch size for encoding
            
        Returns:
            Array of embeddings
        """
        if not self.is_available():
            return np.array([])
        
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = self.encode(batch, use_cache=True)
            all_embeddings.extend(embeddings)
        
        return np.array(all_embeddings)


# Global instance
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Get or create global embedding service instance."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
