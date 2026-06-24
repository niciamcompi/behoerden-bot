"""
ingestion/ingest.py - Embeddings via Ollama (lokal, kostenlos)
"""

import json, os, re, time
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import httpx
from tqdm import tqdm

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "behoerden_ch")
RAW_DIR = Path(__file__).parent.parent / "data" / "raw"

OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
EMBEDDING_DIM = 768
CHUNK_SIZE = 150
CHUNK_OVERLAP = 20
MAX_CHARS = 1500


def embed_text(text: str) -> list[float]:
    text = text[:MAX_CHARS]
    resp = httpx.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=30
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


def chunk_text(text: str) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks, current, cur_len = [], [], 0
    for para in paragraphs:
        words = para.split()
        if cur_len + len(words) > CHUNK_SIZE and current:
            chunks.append(" ".join(current))
            current = current[-CHUNK_OVERLAP:] + words
            cur_len = len(current)
        else:
            current.extend(words)
            cur_len += len(words)
    if current:
        chunks.append(" ".join(current))
    return [c for c in chunks if len(c.split()) > 15]


def iter_pages():
    for f in RAW_DIR.glob("*.json"):
        if not f.name.startswith("_"):
            yield json.loads(f.read_text(encoding="utf-8", errors="ignore"))


def ingest_all():
    # Ollama prüfen
    try:
        httpx.get(f"{OLLAMA_URL}/api/tags", timeout=5).raise_for_status()
        print("Ollama läuft ✓")
    except Exception:
        print("FEHLER: Ollama läuft nicht!")
        return

    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None)
    existing = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION_NAME in existing:
        qdrant.delete_collection(COLLECTION_NAME)
    qdrant.create_collection(COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE))
    print(f"Collection '{COLLECTION_NAME}' erstellt")

    pages = list(iter_pages())
    print(f"{len(pages)} Seiten gefunden\n")

    point_id = 0
    errors = 0
    for page in tqdm(pages, desc="Seiten"):
        for i, chunk in enumerate(chunk_text(page["inhalt"])):
            try:
                emb = embed_text(chunk)
                qdrant.upsert(
                    collection_name=COLLECTION_NAME,
                    points=[PointStruct(
                        id=point_id,
                        vector=emb,
                        payload={
                            "text": chunk,
                            "url": page["url"],
                            "kanton": page["kanton"],
                            "thema": page["thema"],
                            "titel": page["titel"],
                            "quelle_name": page["quelle_name"],
                            "sprache": page.get("sprache", "de"),
                        }
                    )]
                )
                point_id += 1
            except Exception as e:
                errors += 1

    count = qdrant.count(collection_name=COLLECTION_NAME).count
    print(f"\nFERTIG — {count} Chunks in Qdrant ✓ ({errors} Fehler)")


if __name__ == "__main__":
    ingest_all()
