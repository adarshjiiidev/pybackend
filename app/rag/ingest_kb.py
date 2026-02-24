"""
Knowledge Base Ingestion Script — Qdrant Cloud
Run once (or whenever .txt files change) to embed and store all KB docs.

Usage:
    cd backend
    python -m app.rag.ingest_kb
"""

import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> list[str]:
    """Split text into overlapping chunks, preferring paragraph/sentence breaks."""
    if len(text) <= chunk_size:
        return [text.strip()] if text.strip() else []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunk = text[start:].strip()
            if chunk:
                chunks.append(chunk)
            break

        # Prefer paragraph break, then sentence break
        para = text.rfind("\n\n", start, end)
        if para > start + chunk_size // 2:
            end = para
        else:
            sent = max(text.rfind(". ", start, end), text.rfind(".\n", start, end))
            if sent > start + chunk_size // 2:
                end = sent + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap

    return chunks


def ingest():
    """Embed and upsert all KB .txt files into Qdrant Cloud."""
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams, PointStruct
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.error("Run: pip install qdrant-client sentence-transformers")
        sys.exit(1)

    # --- Read credentials directly from .env (avoids importing heavy FastAPI app) ---
    import os
    from dotenv import load_dotenv

    env_path = Path(__file__).parent.parent.parent / ".env"
    load_dotenv(dotenv_path=env_path)

    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    COLLECTION = os.getenv("QDRANT_COLLECTION", "daddys_kb")

    if not QDRANT_URL or not QDRANT_API_KEY:
        logger.error("QDRANT_URL and QDRANT_API_KEY must be set in .env")
        sys.exit(1)

    KB_DIR = Path(__file__).parent.parent.parent / "txt"
    EMBED_MODEL = "all-MiniLM-L6-v2"
    VECTOR_DIM = 384

    if not KB_DIR.exists():
        logger.error(f"KB directory not found: {KB_DIR}")
        sys.exit(1)

    # --- Load embedding model ---
    logger.info(f"🔄 Loading embedding model: {EMBED_MODEL} ...")
    embedder = SentenceTransformer(EMBED_MODEL)
    logger.info("✅ Embedding model loaded")

    # --- Connect to Qdrant Cloud ---
    logger.info(f"🌐 Connecting to Qdrant Cloud: {QDRANT_URL}")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    logger.info("✅ Qdrant Cloud connected")

    # --- Recreate collection ---
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION in existing:
        client.delete_collection(COLLECTION)
        logger.info(f"🗑️  Deleted existing collection '{COLLECTION}'")

    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
    )
    logger.info(f"✅ Created collection '{COLLECTION}' (dim={VECTOR_DIM}, cosine)")

    # --- Process each .txt file ---
    all_points = []
    point_id = 0
    total_files = 0
    txt_files = sorted(KB_DIR.glob("*.txt"))
    logger.info(f"📄 Found {len(txt_files)} .txt files to ingest")

    for file_path in txt_files:
        try:
            content = file_path.read_text(encoding="utf-8")
            if not content.strip():
                logger.warning(f"  ⚠️  Skipping empty file: {file_path.name}")
                continue

            title = file_path.stem.replace("_", " ").title()
            chunks = chunk_text(content)
            if not chunks:
                continue

            logger.info(f"  📝 {file_path.name}: {len(content)} chars → {len(chunks)} chunks")
            embeddings = embedder.encode(chunks, batch_size=32, show_progress_bar=False)

            for chunk_idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                from qdrant_client.models import PointStruct
                all_points.append(
                    PointStruct(
                        id=point_id,
                        vector=embedding.tolist(),
                        payload={
                            "filename": file_path.name,
                            "title": title,
                            "chunk_index": chunk_idx,
                            "total_chunks": len(chunks),
                            "content": chunk,
                            "is_constraints": file_path.name == "constraints.txt",
                        },
                    )
                )
                point_id += 1

            total_files += 1

        except Exception as e:
            logger.error(f"  ❌ Error processing {file_path.name}: {e}")

    # --- Batch upsert to cloud ---
    if all_points:
        BATCH_SIZE = 100
        for i in range(0, len(all_points), BATCH_SIZE):
            batch = all_points[i: i + BATCH_SIZE]
            client.upsert(collection_name=COLLECTION, points=batch)
            logger.info(f"  📤 Upserted batch {i // BATCH_SIZE + 1} ({len(batch)} points)")

        logger.info(f"\n✅ Ingestion complete!")
        logger.info(f"   Files processed : {total_files}/{len(txt_files)}")
        logger.info(f"   Total chunks     : {len(all_points)}")
        logger.info(f"   Collection       : '{COLLECTION}'")
        logger.info(f"   Qdrant Cloud     : {QDRANT_URL}")
    else:
        logger.error("❌ No points to upsert. Check your txt directory.")


if __name__ == "__main__":
    ingest()
