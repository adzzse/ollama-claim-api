import ai_worker


def test_worker_chunk_text():
    text = "First paragraph.\n\nSecond paragraph."
    chunks = ai_worker.chunk_text(text, max_chars=900)
    assert chunks == ["First paragraph.\nSecond paragraph."]
    assert len(chunks) == 1


def test_worker_chunk_text_splits_long_text():
    text = "A" * 500 + "\n\n" + "B" * 500
    chunks = ai_worker.chunk_text(text, max_chars=600)
    assert len(chunks) >= 2


def test_worker_embed_text(monkeypatch):
    def fake_post(url, json, headers, timeout):
        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {"embedding": [0.1, -0.2, 0.35]}

        return FakeResponse()

    monkeypatch.setattr(ai_worker.requests, "post", fake_post)

    result = ai_worker.embed_text("test text")
    assert result == [0.1, -0.2, 0.35]
