"""Laboratorio 3a: crea embeddings e indicizza documenti fittizi in Chroma.
Avvio: python 03_indicizza.py
"""
import json
from rag_common import ROOT, get_resources

items = json.loads((ROOT / "dati" / "kb_demo.json").read_text(encoding="utf-8"))
embed, collection = get_resources()

# Dataset corto: un documento = un chunk. Per documenti lunghi implementare
# chunking per struttura/token e preservare pagina, versione e permessi.
texts = [entry["body"] for entry in items]
embeddings = embed.encode(texts, normalize_embeddings=True).tolist()
ids = [entry["id"] for entry in items]
metadata = [{"doc_id": entry["id"], "title": entry["title"],
             "source": entry["source"], "version": "demo-1"} for entry in items]
collection.upsert(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadata)
print(f"KB indicizzata: {collection.count()} documenti in {ROOT / 'dati' / 'chroma_demo'}")
