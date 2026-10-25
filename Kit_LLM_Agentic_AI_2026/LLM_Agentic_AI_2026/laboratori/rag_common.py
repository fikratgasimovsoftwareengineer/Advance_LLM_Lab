"""Utilità condivise per la KB dimostrativa. Nessun testo è universitario reale."""
from pathlib import Path
import os
import chromadb
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
DB_DIR = Path(os.environ.get("CHROMA_DIR", str(ROOT / "dati" / "chroma_demo")))
COLLECTION_NAME = "campus_demo_italiano"
EMBED_MODEL = os.environ.get("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


def get_resources():
    embed = SentenceTransformer(EMBED_MODEL)
    chroma = chromadb.PersistentClient(path=str(DB_DIR))
    collection = chroma.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    return embed, collection


def find_documents(question, embed, collection, k=3):
    query_vec = embed.encode([question], normalize_embeddings=True).tolist()
    matches = collection.query(query_embeddings=query_vec, n_results=k,
                               include=["documents", "metadatas", "distances"])
    return [
        {"id": meta["doc_id"], "title": meta["title"], "source": meta["source"],
         "text": document, "distance": round(float(distance),4)}
        for meta, document, distance in zip(
            matches["metadatas"][0], matches["documents"][0], matches["distances"][0]
        )
    ]
