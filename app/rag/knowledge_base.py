"""
Enhanced Knowledge Base RAG System
Uses comprehensive index for accurate multi-file retrieval with priority-based ranking.
"""

import os
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path
import re

from .knowledge_index import KNOWLEDGE_INDEX, CATEGORIES, QUICK_LOOKUP

logger = logging.getLogger(__name__)


class KnowledgeBaseRAG:
    """
    Enhanced RAG system with comprehensive structured index.
    Provides accurate retrieval using topic mapping, priorities, and relationships.
    """
    
    def __init__(self, knowledge_base_dir: str = "txt"):
        """Initialize with knowledge base directory and index."""
        self.kb_dir = Path(knowledge_base_dir)
        self.documents = {}
        self.index = KNOWLEDGE_INDEX
        self.quick_lookup = QUICK_LOOKUP
        self._load_documents()
    
    def _load_documents(self):
        """Load all text documents from knowledge base."""
        if not self.kb_dir.exists():
            logger.warning(f"Knowledge base directory not found: {self.kb_dir}")
            return
        
        for file_path in self.kb_dir.glob("*.txt"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    self.documents[file_path.name] = {
                        "content": content,
                        "title": file_path.stem.replace('_', ' ').title()
                    }
                logger.info(f"Loaded: {file_path.name}")
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
        
        logger.info(f"Loaded {len(self.documents)} knowledge base files")
    
    def _match_topics(self, query: str) -> List[str]:
        """Match query to indexed topics using quick lookup and keywords."""
        query_lower = query.lower()
        matched_topics = set()
        
        # Check quick lookup first
        for phrase, topics in self.quick_lookup.items():
            if phrase in query_lower:
                if isinstance(topics, list):
                    matched_topics.update(topics)
                else:
                    matched_topics.add(topics)
        
        # Check keywords in index
        for topic, topic_data in self.index.items():
            for keyword in topic_data.get("keywords", []):
                if keyword in query_lower:
                    matched_topics.add(topic)
                    break
        
        return list(matched_topics)
    
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Search knowledge base using comprehensive index.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of relevant documents with metadata
        """
        # Match topics from query
        matched_topics = self._match_topics(query)
        
        if not matched_topics:
            logger.info(f"No topics matched for query: {query}")
            return []
        
        # Build file list with priorities
        file_scores = {}
        
        for topic in matched_topics:
            if topic not in self.index:
                continue
            
            topic_data = self.index[topic]
            category = topic_data.get("category", "")
            priority = CATEGORIES.get(category, {}).get("priority", 3)
            
            # Primary files get higher score
            for filename in topic_data.get("primary_files", []):
                file_scores[filename] = file_scores.get(filename, 0) + (10 - priority) * 3
            
            # Related files get lower score
            for filename in topic_data.get("related_files", []):
                file_scores[filename] = file_scores.get(filename, 0) + (10 - priority)
        
        # ALWAYS include constraints.txt first
        if "constraints.txt" not in file_scores and "constraints.txt" in self.documents:
            file_scores["constraints.txt"] = 100  # Highest priority
        
        # Sort by score and get top_k
        sorted_files = sorted(file_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # Build results
        results = []
        for filename, score in sorted_files:
            if filename in self.documents:
                results.append({
                    "filename": filename,
                    "title": self.documents[filename]["title"],
                    "content": self.documents[filename]["content"],
                    "score": score,
                    "topics": [t for t in matched_topics if filename in self.index.get(t, {}).get("primary_files", []) + self.index.get(t, {}).get("related_files", [])]
                })
        
        logger.info(f"Retrieved {len(results)} files for topics: {matched_topics}")
        return results
    
    def get_relevant_context(self, query: str, max_chars: int = 3000) -> str:
        """
        Get formatted context from knowledge base for a query.
        
        Args:
            query: User query
            max_chars: Maximum characters to return
            
        Returns:
            Formatted context string with multiple relevant documents
        """
        results = self.search(query, top_k=3)
        
        if not results:
            return ""
        
        context_parts = []
        total_chars = 0
        
        # ALWAYS include constraints if present
        constraints_result = next((r for r in results if r["filename"] == "constraints.txt"), None)
        if constraints_result:
            header = f"\n## CRITICAL RULES (constraints.txt)\n"
            content = constraints_result['content'][:800] + "..." if len(constraints_result['content']) > 800 else constraints_result['content']
            context_parts.append(header + content)
            total_chars += len(header) + len(content)
            results.remove(constraints_result)
        
        # Add other results
        for result in results:
            if total_chars >= max_chars:
                break
            
            header = f"\n## {result['title']} ({result['filename']})\n"
            available = max_chars - total_chars - len(header)
            
            if available <= 0:
                break
            
            content = result['content']
            if len(content) > available:
                content = content[:available] + "..."
            
            context_parts.append(header + content)
            total_chars += len(header) + len(content)
        
        if context_parts:
            return "\n---\n**Knowledge Base Context:**\n" + "\n".join(context_parts)
        return ""
    
    def list_documents(self) -> List[str]:
        """List all available documents."""
        return [doc["title"] for doc in self.documents.values()]
    
    def get_topics(self) -> List[str]:
        """Get all indexed topics."""
        return list(self.index.keys())


# Global instance  
_kb_rag: Optional[KnowledgeBaseRAG] = None


def get_kb_rag() -> KnowledgeBaseRAG:
    """Get or create global knowledge base RAG instance."""
    global _kb_rag
    if _kb_rag is None:
        _kb_rag = KnowledgeBaseRAG()
    return _kb_rag


# Tool function for agents
async def search_knowledge_base(query: str) -> Dict[str, Any]:
    """
    Search knowledge base for relevant information.
    Used by agents to retrieve domain-specific knowledge.
    """
    try:
        kb = get_kb_rag()
        results = kb.search(query, top_k=2)
        
        if not results:
            return {
                "found": False,
                "message": "No relevant information found in knowledge base"
            }
        
        return {
            "found": True,
            "results": [
                {
                    "title": r["title"],
                    "filename": r["filename"],
                    "topics": r.get("topics", []),
                    "preview": r["content"][:600] + "..." if len(r["content"]) > 600 else r["content"]
                }
                for r in results
            ],
            "matched_topics": list(set([t for r in results for t in r.get("topics", [])]))
        }
    except Exception as e:
        logger.error(f"Knowledge base search error: {e}")
        return {"found": False, "error": str(e)}
