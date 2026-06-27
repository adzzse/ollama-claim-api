import asyncio
import json
import logging
import os
import tempfile
import uuid

import fitz
from minio import Minio
import pika

from app.ollama_client import generate_embeddings
from app.settings import load_settings

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


def extract_pdf_text(filepath: str) -> str:
    doc = fitz.open(filepath)
    try:
        pages = [page.get_text("text") for page in doc]
    finally:
        doc.close()
    return "\n\n".join(pages).strip()


def chunk_document_text(text: str, max_chars: int = 900) -> list[str]:
    logger.debug("chunk text start text_chars=%s max_chars=%s", len(text), max_chars)
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
    result = chunks or [text[:max_chars]]
    logger.debug("chunk text complete paragraphs=%s chunks=%s", len(paragraphs), len(result))
    return result


async def build_chunks(document_id: str, text: str, settings) -> dict:
    raw_chunks = chunk_document_text(text)
    chunks = []
    for i, chunk_text in enumerate(raw_chunks):
        embedding = await generate_embeddings(chunk_text, settings)
        chunks.append({
            "chunkId": str(uuid.uuid4()),
            "chunkIndex": i + 1,
            "text": chunk_text,
            "embedding": embedding,
        })
    return {"documentId": document_id, "chunks": chunks}


def main():
    settings = load_settings()
    logger.info(
        "Settings loaded: ollama=%s model=%s embed_model=%s",
        settings.ollama_base_url,
        settings.ollama_model,
        settings.ollama_embedding_model,
    )

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

    connection = pika.BlockingConnection(
        pika.ConnectionParameters(
            host=RABBITMQ_HOST,
            port=RABBITMQ_PORT,
            credentials=pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS),
            heartbeat=0,
        )
    )
    channel = connection.channel()
    channel.queue_declare(queue=RABBITMQ_INPUT_QUEUE, durable=True)
    channel.queue_declare(queue=RABBITMQ_OUTPUT_QUEUE, durable=True)
    logger.info("Connected to RabbitMQ on %s:%s", RABBITMQ_HOST, RABBITMQ_PORT)

    def on_message(ch, method, properties, body):
        tmp_path = None
        try:
            document_id = body.decode('utf-8').strip().strip('"')
            logging.info(f"Received document: {document_id}")
            if not document_id:
                logger.error("No documentId in message: %s", body)
                ch.basic_nack(method.delivery_tag, requeue=False)
                return

            pdf_object = f"sources/raw/{document_id}.pdf"
            tmp_path = os.path.join(tempfile.gettempdir(), f"{document_id}.pdf")

            logger.info("Processing documentId=%s", document_id)

            minio_client.fget_object(MINIO_BUCKET, pdf_object, tmp_path)
            logger.info("Downloaded %s from MinIO bucket %s", pdf_object, MINIO_BUCKET)

            text = extract_pdf_text(tmp_path)
            logger.info("Extracted %d chars from PDF", len(text))

            result = asyncio.run(build_chunks(document_id, text, settings))
            result_json = json.dumps(result)
            logger.info(
                "Built %d chunks for documentId=%s",
                len(result["chunks"]),
                document_id,
            )

            channel.basic_publish(
                exchange="",
                routing_key=RABBITMQ_OUTPUT_QUEUE,
                body=result_json,
                properties=pika.BasicProperties(
                    content_type="application/json",
                    delivery_mode=2,
                ),
            )
            logger.info("Published result for documentId=%s", document_id)

            ch.basic_ack(method.delivery_tag)
            logger.info("Ack'd message for documentId=%s", document_id)

        except Exception:
            logger.exception("Failed to process message")
            try:
                ch.basic_nack(method.delivery_tag, requeue=False)
            except Exception:
                logger.exception("Failed to nack message")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                    logger.debug("Cleaned up temp file %s", tmp_path)
                except Exception:
                    logger.exception("Failed to clean up temp file %s", tmp_path)

    channel.basic_consume(
        queue=RABBITMQ_INPUT_QUEUE,
        on_message_callback=on_message,
        auto_ack=False,
    )
    logger.info("Waiting for messages on %s...", RABBITMQ_INPUT_QUEUE)
    channel.start_consuming()


if __name__ == "__main__":
    main()
