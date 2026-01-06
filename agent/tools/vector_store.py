import os
import logging
from typing import List, Dict, Any, Optional
from pinecone import Pinecone, ServerlessSpec
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

logger = logging.getLogger("financial_agent")

class VectorStoreTool:
    """
    Tool to interact with Pinecone for storing and retrieving financial knowledge.
    """
    
    def __init__(self):
        self.api_key = os.getenv("PINECONE_API_KEY")
        self.index_name = os.getenv("PINECONE_INDEX")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        
        if not self.api_key or not self.index_name:
            logger.warning("Pinecone credentials not found. Vector memory will be disabled.")
            self.pc = None
            self.index = None
            self.embeddings = None
            return

        try:
            self.pc = Pinecone(api_key=self.api_key)
            self.index = self.pc.Index(self.index_name)
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001", 
                google_api_key=self.gemini_api_key
            )
        except Exception as e:
            logger.warning(f"Failed to initialize Pinecone: {e}")
            self.pc = None
            self.index = None

    def upsert_documents(self, documents: List[Document]):
        """
        Embeds and upserts documents to Pinecone.
        """
        if not self.index or not self.embeddings:
            return

        try:
            texts = [d.page_content for d in documents]
            metadatas = [d.metadata for d in documents]
            
            # Embed documents
            vectors = self.embeddings.embed_documents(texts)
            
            # Prepare for upsert
            to_upsert = []
            for i, (text, meta, vec) in enumerate(zip(texts, metadatas, vectors)):
                # Create a unique ID. Ideally hashed content or UUID.
                # Here we use a simple generic ID for demo, but better to use hash of content
                import hashlib
                doc_id = hashlib.md5(text.encode()).hexdigest()
                to_upsert.append((doc_id, vec, {**meta, "text": text}))
            
            # Upsert in batches if needed, Pinecone handles some list sizes
            self.index.upsert(vectors=to_upsert)
            logger.info(f"Upserted {len(to_upsert)} documents to Pinecone.")

        except Exception as e:
            logger.error(f"Error performing upsert: {e}")

    def similarity_search(self, query: str, k: int = 3, filter: Optional[Dict] = None) -> List[Document]:
        """
        Performs semantic search.
        """
        if not self.index or not self.embeddings:
            return []

        try:
            query_vector = self.embeddings.embed_query(query)
            results = self.index.query(
                vector=query_vector, 
                top_k=k, 
                include_metadata=True,
                filter=filter
            )
            
            docs = []
            for match in results.matches:
                text = match.metadata.get("text", "")
                # remove text from metadata to avoid duplication in Document
                meta = {k: v for k, v in match.metadata.items() if k != "text"}
                docs.append(Document(page_content=text, metadata=meta))
            
            return docs

        except Exception as e:
            logger.error(f"Error during similarity search: {e}")
            return []
