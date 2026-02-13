"""
Metadata Indexer for Knowledge Base
Extracts and indexes entities, concepts, and topics from knowledge files.
"""

import logging
import re
from typing import Dict, List, Set
from pathlib import Path

logger = logging.getLogger(__name__)


class MetadataIndexer:
    """Extract and index metadata from knowledge base files."""
    
    # Define extraction patterns
    SYMBOL_PATTERNS = [
        r'\b([A-Z]{2,5})\b',  # Stock symbols (2-5 uppercase letters)
        r'\b(NIFTY|BANKNIFTY|SENSEX)\b',  # Index names
    ]
    
    # Known concepts from knowledge base
    KNOWN_CONCEPTS = {
        "coa": ["COA 1.0", "COA 2.0", "Chart of Account", "Chandmama Approach"],
        "ltp": ["LTP", "Last Traded Price", "LTP Calculator"],
        "soc": ["SOC", "Strong of Candles", "Support Resistance"],
        "options": ["Option Chain", "Call", "Put", "Strike Price", "Premium"],
        "portfolio": ["Portfolio", "Allocation", "Diversification", "Asset Allocation"],
        "technical": ["Support", "Resistance", "Breakout", "Trend", "Moving Average"],
        "trading": ["Entry", "Exit", "Stop Loss", "Target", "Risk Management"],
    }
    
    def __init__(self):
        """Initialize metadata indexer."""
        self.metadata_cache: Dict[str, Dict] = {}
    
    def extract_symbols(self, text: str) -> Set[str]:
        """
        Extract stock symbols and indices from text.
        
        Args:
            text: Text to extract from
            
        Returns:
            Set of symbol strings
        """
        symbols = set()
        
        for pattern in self.SYMBOL_PATTERNS:
            matches = re.findall(pattern, text)
            symbols.update(matches)
        
        # Filter out common words that aren't symbols
        false_positives = {"THE", "AND", "FOR", "ARE", "NOT", "BUT", "CAN", "ALL", "YOU", "THIS"}
        symbols = {s for s in symbols if s not in false_positives}
        
        return symbols
    
    def extract_concepts(self, text: str, filename: str = "") -> List[str]:
        """
        Extract trading/financial concepts from text.
        
        Args:
            text: Text to analyze
            filename: Optional filename for context
            
        Returns:
            List of concept identifiers
        """
        text_lower = text.lower()
        concepts = []
        
        # Check filename first for concept hints
        filename_lower = filename.lower()
        for concept_key, concept_terms in self.KNOWN_CONCEPTS.items():
            if concept_key in filename_lower:
                concepts.append(concept_key)
                continue
            
            # Check if any concept term appears in text
            for term in concept_terms:
                if term.lower() in text_lower:
                    concepts.append(concept_key)
                    break
        
        return list(set(concepts))
    
    def categorize_document(self, filename: str, content: str) -> str:
        """
        Categorize document based on filename and content.
        
        Returns:
            Category string: 'trading_strategy', 'technical_analysis', 'options', etc.
        """
        filename_lower = filename.lower()
        content_lower = content.lower()
        
        # Filename-based categorization
        if any(x in filename_lower for x in ["coa", "soc", "wtb", "wtt", "strong"]):
            return "trading_strategy"
        elif any(x in filename_lower for x in ["option", "chain"]):
            return "options_trading"
        elif any(x in filename_lower for x in ["ltp", "calculator"]):
            return "tools"
        elif any(x in filename_lower for x in ["support", "resistance", "technical"]):
            return "technical_analysis"
        elif "portfolio" in filename_lower:
            return "portfolio_management"
        elif "about" in filename_lower:
            return "meta"
        
        # Content-based fallback
        if "option" in content_lower and "strike" in content_lower:
            return "options_trading"
        elif any(x in content_lower for x in ["support", "resistance", "trend"]):
            return "technical_analysis"
        
        return "general_finance"
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities (people, places, organizations).
        
        Returns:
            Dict with entity types as keys
        """
        entities = {
            "people": [],
            "organizations": [],
            "locations": []
        }
        
        # Known entities from knowledge base
        if "vinay" in text.lower() or "vinay prakash tiwari" in text.lower():
            entities["people"].append("Vinay Prakash Tiwari")
        
        if "adarsh" in text.lower():
            entities["people"].append("Adarsh")
        
        if "investingdaddy" in text.lower():
            entities["organizations"].append("InvestingDaddy")
        
        if "daddy's international school" in text.lower() or "chandauli" in text.lower():
            entities["locations"].append("Chandauli")
        
        return {k: v for k, v in entities.items() if v}  # Remove empty lists
    
    def index_document(self, filename: str, content: str) -> Dict:
        """
        Full metadata extraction for a document.
        
        Args:
            filename: Document filename
            content: Document content
            
        Returns:
            Metadata dictionary
        """
        metadata = {
            "filename": filename,
            "symbols": list(self.extract_symbols(content)),
            "concepts": self.extract_concepts(content, filename),
            "category": self.categorize_document(filename, content),
            "entities": self.extract_entities(content),
            "word_count": len(content.split()),
            "char_count": len(content)
        }
        
        # Cache metadata
        self.metadata_cache[filename] = metadata
        
        logger.debug(f"Indexed {filename}: {len(metadata['symbols'])} symbols, {len(metadata['concepts'])} concepts")
        
        return metadata
    
    def get_files_by_symbol(self, symbol: str) -> List[str]:
        """Get all files mentioning a specific symbol."""
        return [
            filename
            for filename, meta in self.metadata_cache.items()
            if symbol.upper() in [s.upper() for s in meta.get("symbols", [])]
        ]
    
    def get_files_by_concept(self, concept: str) -> List[str]:
        """Get all files related to a specific concept."""
        concept_lower = concept.lower()
        return [
            filename
            for filename, meta in self.metadata_cache.items()
            if concept_lower in meta.get("concepts", [])
        ]
    
    def get_files_by_category(self, category: str) -> List[str]:
        """Get all files in a category."""
        return [
            filename
            for filename, meta in self.metadata_cache.items()
            if meta.get("category") == category
        ]
    
    def get_metadata(self, filename: str) -> Dict:
        """Get metadata for a specific file."""
        return self.metadata_cache.get(filename, {})
    
    def get_all_symbols(self) -> Set[str]:
        """Get all symbols across all documents."""
        symbols = set()
        for meta in self.metadata_cache.values():
            symbols.update(meta.get("symbols", []))
        return symbols
    
    def get_all_concepts(self) -> Set[str]:
        """Get all concepts across all documents."""
        concepts = set()
        for meta in self.metadata_cache.values():
            concepts.update(meta.get("concepts", []))
        return concepts


# Global instance
_metadata_indexer: MetadataIndexer = None


def get_metadata_indexer() -> MetadataIndexer:
    """Get or create global metadata indexer."""
    global _metadata_indexer
    if _metadata_indexer is None:
        _metadata_indexer = MetadataIndexer()
    return _metadata_indexer
