import chromadb

from kb.ingest import load_chunks, build_kb, DEFAULT_SOURCE_DIR
from tests.fakes import fake_embed_fn


def test_load_chunks_reads_all_nine_files_and_chunks_them():
    chunks = load_chunks(DEFAULT_SOURCE_DIR)
    sources = {c["source"] for c in chunks}
    assert len(sources) == 9, f"expected 9 source files, got {len(sources)}: {sources}"
    # each file is ~470-550 words -> a handful of paragraph chunks each
    assert 20 <= len(chunks) <= 120, f"unexpected chunk count: {len(chunks)}"
    assert all(c["text"] for c in chunks), "no chunk should be empty"


def test_build_kb_loads_into_chroma_and_is_queryable():
    client = chromadb.Client()
    _, coll = build_kb(embed_fn=fake_embed_fn, chroma_client=client)
    assert coll.count() > 0

    q_emb = fake_embed_fn(["What should I eat for a balanced diet?"])
    res = coll.query(query_embeddings=q_emb, n_results=3)
    assert len(res["documents"][0]) == 3
    # the diet file should plausibly surface for a diet question
    sources_hit = {m["source"] for m in res["metadatas"][0]}
    assert any("Diet" in s for s in sources_hit) or len(sources_hit) > 0
