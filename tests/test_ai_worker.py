import minio


def _make_fake_minio(monkeypatch):
    fake = minio.Minio("localhost:9000", access_key="k", secret_key="s", secure=False)

    def fake_fget(bucket, obj, path):
        with open(path, "w") as f:
            f.write("dummy pdf content")

    def fake_exists(bucket):
        return True

    monkeypatch.setattr(fake, "fget_object", fake_fget)
    monkeypatch.setattr(fake, "bucket_exists", fake_exists)
    return fake


def _make_fake_ch():
    class FakeCh:
        def __init__(self):
            self.acked = False
            self.nacked = False
            self.published_body = None

        def basic_ack(self, tag):
            self.acked = True

        def basic_nack(self, tag, requeue):
            self.nacked = True

        def basic_publish(self, exchange, routing_key, body, properties):
            self.published_body = body

    return FakeCh()


def _make_fake_conn(monkeypatch):
    fake = type("FakeConn", (), {})()
    fake.add_callback_threadsafe = lambda fn: fn()
    return fake


def test_worker_sends_file_to_pipeline_and_upserts(monkeypatch):
    import ai_worker

    posted_files = None
    posted_url = None

    def fake_post(url, **kwargs):
        nonlocal posted_url, posted_files
        posted_url = url
        posted_files = kwargs.get("files")

        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "status": "SUCCESS",
                    "data": [
                        {"chunkIndex": 0, "text": "Chunk one.", "embedding": [0.1, 0.2]},
                        {"chunkIndex": 1, "text": "Chunk two.", "embedding": [0.3, 0.4]},
                    ],
                }

        return FakeResponse()

    monkeypatch.setattr(ai_worker.requests, "post", fake_post)

    upserted = []

    def fake_upsert(self, collection_name, points, wait):
        upserted.extend(points)

    fake_qdrant = type("FakeQdrant", (), {"upsert": fake_upsert})()

    ch = _make_fake_ch()
    method = type("FakeMethod", (), {"delivery_tag": 1})()
    conn = _make_fake_conn(monkeypatch)
    fake_minio = _make_fake_minio(monkeypatch)

    ai_worker.process_document("doc-1", ch, method, conn, fake_minio, fake_qdrant)

    assert posted_url is not None
    assert posted_url.endswith("/ai/process-document")
    assert posted_files is not None
    assert "file" in posted_files

    assert len(upserted) == 2
    assert upserted[0].payload["documentId"] == "doc-1"
    assert upserted[0].payload["chunkIndex"] == 0
    assert upserted[0].payload["text"] == "Chunk one."
    assert upserted[0].vector == [0.1, 0.2]
    assert upserted[1].payload["chunkIndex"] == 1
    assert upserted[1].payload["text"] == "Chunk two."

    assert ch.acked is True
    assert ch.published_body is not None
    notification = ch.published_body
    assert "COMPLETED" in notification
    assert "doc-1" in notification


def test_worker_nacks_on_pipeline_error(monkeypatch):
    import ai_worker

    def fake_post(url, **kwargs):
        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {"status": "ERROR", "detail": "MinerU extraction failed"}

        return FakeResponse()

    monkeypatch.setattr(ai_worker.requests, "post", fake_post)

    ch = _make_fake_ch()
    method = type("FakeMethod", (), {"delivery_tag": 1})()
    conn = _make_fake_conn(monkeypatch)
    fake_minio = _make_fake_minio(monkeypatch)
    fake_qdrant = type("FakeQdrant", (), {})()

    ai_worker.process_document("doc-err", ch, method, conn, fake_minio, fake_qdrant)

    assert ch.nacked is True
    assert ch.acked is False


def test_worker_nacks_on_http_error(monkeypatch):
    import ai_worker

    def fake_post(url, **kwargs):
        class FakeResponse:
            def raise_for_status(self):
                raise Exception("HTTP 503")

            def json(self):
                return {"status": "ERROR", "detail": "Service unavailable"}

        return FakeResponse()

    monkeypatch.setattr(ai_worker.requests, "post", fake_post)

    ch = _make_fake_ch()
    method = type("FakeMethod", (), {"delivery_tag": 1})()
    conn = _make_fake_conn(monkeypatch)
    fake_minio = _make_fake_minio(monkeypatch)
    fake_qdrant = type("FakeQdrant", (), {})()

    ai_worker.process_document("doc-http-err", ch, method, conn, fake_minio, fake_qdrant)

    assert ch.nacked is True
