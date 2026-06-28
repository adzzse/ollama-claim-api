import json
import logging
import os
import tempfile
import threading
import uuid

import fitz
from minio import Minio
import pika
import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ai_worker")

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "MinioAdminSecurePass123!")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "evidence-pilot-bucket")

RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "localhost")
RABBITMQ_PORT = int(os.getenv("RABBITMQ_PORT", "5672"))
RABBITMQ_USER = os.getenv("RABBITMQ_USER", "rabbitmqadmin")
RABBITMQ_PASS = os.getenv("RABBITMQ_PASS", "RabbitMqAdminPass123!")
RABBITMQ_INPUT_QUEUE = os.getenv("RABBITMQ_INPUT_QUEUE", "extraction.queue")
RABBITMQ_OUTPUT_QUEUE = os.getenv("RABBITMQ_OUTPUT_QUEUE", "extraction.result.queue")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "https://good-lumpish-headstone.ngrok-free.dev")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = "evidence_chunks"
EMBEDDING_DIM = 768


def extract_pdf_text(filepath: str) -> str:
    doc = fitz.open(filepath)
    try:
        pages = [page.get_text("text") for page in doc]
    finally:
        doc.close()
    return "\n\n".join(pages).strip()


def chunk_text(text: str, max_chars: int = 900) -> list[str]:
    paragraphs = [part.strip() for part in text.split("\n") if part.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = f"{current}\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = paragraph[:max_chars]
    if current:
        chunks.append(current)
    return chunks or [text[:max_chars]]


HEADERS = {"ngrok-skip-browser-warning": "true"}


def embed_text(text: str) -> list[float]:
    response = requests.post(
        f"{OLLAMA_BASE_URL}/ai/embeddings",
        json={"model": OLLAMA_EMBEDDING_MODEL, "prompt": text},
        headers=HEADERS,
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    embedding = data.get("embedding")
    if embedding is not None:
        return embedding
    embeddings = data.get("embeddings")
    if embeddings and isinstance(embeddings, list):
        return embeddings[0]
    raise ValueError("No embedding vector in Ollama response")


def process_document(
    document_id: str,
    ch: pika.adapters.blocking_connection.BlockingChannel,
    method: pika.spec.Basic.Deliver,
    connection: pika.BlockingConnection,
    minio_client: Minio,
    qdrant_client: QdrantClient,
) -> None:
    tmp_path = None
    try:
        pdf_object = f"sources/raw/{document_id}.pdf"
        tmp_path = os.path.join(tempfile.gettempdir(), f"{document_id}.pdf")

        logger.info("Processing documentId=%s", document_id)

        minio_client.fget_object(MINIO_BUCKET, pdf_object, tmp_path)
        logger.info("Downloaded %s from MinIO", pdf_object)

        text = extract_pdf_text(tmp_path)
        logger.info("Extracted %d chars from PDF", len(text))

        raw_chunks = chunk_text(text)
        logger.info("Chunked into %d chunks", len(raw_chunks))

        points: list[PointStruct] = []
        for i, chunk_text_val in enumerate(raw_chunks):
            embedding = embed_text(chunk_text_val)
            point_id = str(uuid.uuid4())
            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "documentId": document_id,
                        "chunkIndex": i + 1,
                        "text": chunk_text_val,
                    },
                )
            )
            logger.debug(
                "Embedded chunk %d/%d for documentId=%s",
                i + 1,
                len(raw_chunks),
                document_id,
            )

        qdrant_client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=points,
            wait=True,
        )
        logger.info(
            "Upserted %d chunks to Qdrant for documentId=%s",
            len(points),
            document_id,
        )

        def ack_and_notify():
            try:
                ch.basic_ack(method.delivery_tag)
                logger.info("Ack'd message for documentId=%s", document_id)
            except Exception:
                logger.exception("Failed to ack for documentId=%s", document_id)

            try:
                notification = json.dumps({
                    "documentId": document_id,
                    "status": "COMPLETED",
                    "totalChunks": len(points),
                })
                ch.basic_publish(
                    exchange="",
                    routing_key=RABBITMQ_OUTPUT_QUEUE,
                    body=notification,
                    properties=pika.BasicProperties(
                        content_type="application/json",
                        delivery_mode=2,
                    ),
                )
                logger.info(
                    "Published completion for documentId=%s totalChunks=%s",
                    document_id,
                    len(points),
                )
            except Exception:
                logger.exception("Failed to publish notification for documentId=%s", document_id)

        connection.add_callback_threadsafe(ack_and_notify)

    except Exception:
        logger.exception("Failed to process documentId=%s", document_id)

        def nack_message():
            try:
                ch.basic_nack(method.delivery_tag, requeue=False)
                logger.info("Nack'd message for documentId=%s (sent to DLQ)", document_id)
            except Exception:
                logger.exception("Failed to nack for documentId=%s", document_id)

        connection.add_callback_threadsafe(nack_message)

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
                logger.debug("Cleaned up temp file %s", tmp_path)
            except Exception:
                logger.exception("Failed to clean up temp file %s", tmp_path)


def main():
    minio_client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False,
    )
    logger.info("MinIO client connected to %s", MINIO_ENDPOINT)

    if not minio_client.bucket_exists(MINIO_BUCKET):
        minio_client.make_bucket(MINIO_BUCKET)
        logger.info("Created bucket %s", MINIO_BUCKET)

    qdrant_client = QdrantClient(url=QDRANT_URL)
    logger.info("Qdrant client connected to %s", QDRANT_URL)

    existing_collections = [
        c.name for c in qdrant_client.get_collections().collections
    ]
    if QDRANT_COLLECTION not in existing_collections:
        qdrant_client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        logger.info("Created Qdrant collection %s", QDRANT_COLLECTION)
    else:
        logger.info("Qdrant collection %s already exists", QDRANT_COLLECTION)

    connection = pika.BlockingConnection(
        pika.ConnectionParameters(
            host=RABBITMQ_HOST,
            port=RABBITMQ_PORT,
            credentials=pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS),
        )
    )
    channel = connection.channel()
    channel.queue_declare(queue=RABBITMQ_INPUT_QUEUE, durable=True)
    channel.queue_declare(queue=RABBITMQ_OUTPUT_QUEUE, durable=True)
    channel.basic_qos(prefetch_count=1)
    logger.info("Connected to RabbitMQ on %s:%s", RABBITMQ_HOST, RABBITMQ_PORT)

    def on_message(ch, method, properties, body):
        document_id = body.decode("utf-8").strip().strip('"')
        if not document_id:
            logger.error("Empty documentId in message: %s", body)
            ch.basic_nack(method.delivery_tag, requeue=False)
            return

        logger.info("Received documentId=%s, spawning background thread", document_id)

        t = threading.Thread(
            target=process_document,
            args=(
                document_id,
                ch,
                method,
                connection,
                minio_client,
                qdrant_client,
            ),
            daemon=True,
        )
        t.start()

    channel.basic_consume(
        queue=RABBITMQ_INPUT_QUEUE,
        on_message_callback=on_message,
        auto_ack=False,
    )
    logger.info("Waiting for messages on %s...", RABBITMQ_INPUT_QUEUE)
    channel.start_consuming()


if __name__ == "__main__":
    main()
