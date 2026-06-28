import os

import pika

RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "localhost")
RABBITMQ_PORT = int(os.getenv("RABBITMQ_PORT", "5672"))
RABBITMQ_USER = os.getenv("RABBITMQ_USER", "rabbitmqadmin")
RABBITMQ_PASS = os.getenv("RABBITMQ_PASS", "RabbitMqAdminPass123!")
RABBITMQ_INPUT_QUEUE = os.getenv("RABBITMQ_INPUT_QUEUE", "extraction.queue")


def trigger_ingestion(document_id: str) -> None:
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(
            host=RABBITMQ_HOST,
            port=RABBITMQ_PORT,
            credentials=pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS),
        )
    )
    channel = connection.channel()

    channel.queue_declare(queue=RABBITMQ_INPUT_QUEUE, durable=True)

    channel.basic_publish(
        exchange="",
        routing_key=RABBITMQ_INPUT_QUEUE,
        body=document_id,
        properties=pika.BasicProperties(delivery_mode=2),
    )
    print(f"Successfully published test documentId: {document_id}")
    connection.close()

if __name__ == "__main__":
    # Pass a valid or dummy UUID that exists in your MinIO test bucket
    test_uuid = "862db573-6744-4523-8f2f-a6e141215388"
    trigger_ingestion(test_uuid)