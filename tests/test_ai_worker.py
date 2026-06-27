import asyncio


def test_worker_builds_chunks_without_demo_store(monkeypatch):
    import ai_worker

    async def fake_generate_embeddings(text, settings):
        return [float(len(text))]

    monkeypatch.setattr(ai_worker, "generate_embeddings", fake_generate_embeddings)

    text = "First paragraph.\n\nSecond paragraph."
    result = asyncio.run(ai_worker.build_chunks("doc-1", text, object()))

    assert result["documentId"] == "doc-1"
    assert [chunk["chunkIndex"] for chunk in result["chunks"]] == [1]
    assert result["chunks"][0]["text"] == "First paragraph.\nSecond paragraph."
    assert result["chunks"][0]["embedding"] == [34.0]
