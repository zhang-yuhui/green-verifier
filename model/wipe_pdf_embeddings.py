# wipe_qdrant_collections.py
import os, sys
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# If your config is elsewhere, adjust or hardcode host/port.
sys.path.append(os.path.dirname(os.path.dirname(__file__))) if "__file__" in globals() else None

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

# Collections to wipe & recreate
TEXT_COL    = "reports_text"
KPI_COL     = "reports_kpi"

# Use same embedding model as your pipeline for correct dim (384 for MiniLM-L6-v2)
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

def main():
    qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    dim = SentenceTransformer(EMBED_MODEL_NAME).get_sentence_embedding_dimension()

    cols = (TEXT_COL, KPI_COL)

    # Drop if exists
    for col in cols:
        try:
            qdrant.delete_collection(col)
            print(f"Deleted collection: {col}")
        except Exception:
            print(f"(Skip) Collection {col} did not exist or was already removed.")

    # Recreate clean collections
    for col in cols:
        qdrant.recreate_collection(
            collection_name=col,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        print(f"Recreated empty collection: {col} (dim={dim}, distance=cosine)")

if __name__ == "__main__":
    main()
