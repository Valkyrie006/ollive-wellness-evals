"""Chunk the 9 provided wellness .md files, embed them, and load them into an
in-memory Chroma collection (see docs/DESIGN.md - no persistence needed at
this corpus size).

The embedder is injected (`embed_fn`) rather than hard-imported at module
load time so this module can be unit-tested without downloading model
weights (see tests/test_kb_ingest.py). In real use, `default_embed_fn`
(sentence-transformers, all-MiniLM-L6-v2) is what actually runs.
"""
from __future__ import annotations

import glob
import os
from collections.abc import Callable

DEFAULT_SOURCE_DIR = os.path.join(os.path.dirname(__file__), "source")
COLLECTION_NAME = "wellness_kb"


def load_chunks(source_dir: str = DEFAULT_SOURCE_DIR) -> list[dict]:
    """Split each .md file on blank-line paragraph breaks. The 9 provided
    files are ~470-550 words each, so this yields roughly 3-5 chunks per
    file (~30-40 total) - plenty granular for lookup_kb without any need
    for token-count-based splitting.
    """
    chunks = []
    for path in sorted(glob.glob(os.path.join(source_dir, "*.md"))):
        text = open(path, encoding="utf-8").read()
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        for i, para in enumerate(paragraphs):
            chunks.append({
                "id": f"{os.path.basename(path)}-{i}",
                "text": para,
                "source": os.path.basename(path),
            })
    return chunks


def default_embed_fn(texts: list[str]) -> list[list[float]]:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model.encode(texts).tolist()


def build_kb(
    source_dir: str = DEFAULT_SOURCE_DIR,
    embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
    chroma_client=None,
):
    """Returns (chroma_client, collection). `embed_fn` and `chroma_client`
    are injectable for testing; both default to the real implementations.
    """
    if embed_fn is None:
        embed_fn = default_embed_fn
    if chroma_client is None:
        import chromadb
        chroma_client = chromadb.Client()

    chunks = load_chunks(source_dir)
    if not chunks:
        raise ValueError(f"No .md files found in {source_dir}")

    embeddings = embed_fn([c["text"] for c in chunks])
    coll = chroma_client.get_or_create_collection(COLLECTION_NAME)
    coll.add(
        ids=[c["id"] for c in chunks],
        embeddings=embeddings,
        documents=[c["text"] for c in chunks],
        metadatas=[{"source": c["source"]} for c in chunks],
    )
    return chroma_client, coll


if __name__ == "__main__":
    client, coll = build_kb()
    print(f"Loaded {coll.count()} chunks into '{COLLECTION_NAME}'")
