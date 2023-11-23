# stdlib
from collections import defaultdict
import random
from time import sleep

# third party
from faker import Faker
from zmq import Socket

# syft absolute
import syft
from syft.service.queue.base_queue import AbstractMessageHandler
from syft.service.queue.queue import QueueManager
from syft.service.queue.zmq_queue import ZMQClient
from syft.service.queue.zmq_queue import ZMQClientConfig
from syft.service.queue.zmq_queue import ZMQConsumer
from syft.service.queue.zmq_queue import ZMQProducer
from syft.service.queue.zmq_queue import ZMQQueueConfig
from syft.service.response import SyftError
from syft.service.response import SyftSuccess


def test_zmq_client():
    hostname = "127.0.0.1"

    config = ZMQClientConfig(hostname=hostname)

    assert config.hostname == hostname

    client = ZMQClient(config=config)

    assert client.host == hostname
    assert len(client.producers) == 0
    assert len(client.consumers) == 0

    assert isinstance(client.producers, dict)
    assert len(client.producers) == 0

    assert isinstance(client.consumers, defaultdict)
    assert len(client.consumers) == 0

    QueueName = "QueueName"

    producer = client.add_producer(queue_name=QueueName)
    assert isinstance(producer, ZMQProducer)
    assert producer.address is not None
    assert producer.alive
    assert producer.queue_name == QueueName

    received_message = []

    class MyMessageHandler(AbstractMessageHandler):
        queue_name = QueueName

        @staticmethod
        def handle_message(message: bytes):
            received_message.append(message)

    consumer = client.add_consumer(
        queue_name=QueueName, message_handler=MyMessageHandler
    )

    consumer.run()
    # stdlib
    from time import sleep

    sleep(1)
    assert isinstance(consumer, ZMQConsumer)
    assert consumer.address is not None
    assert consumer.alive
    assert consumer.queue_name == QueueName
    assert consumer.address == producer.address

    assert len(client.producers) == 1
    assert len(client.consumers) == 1
    assert QueueName in client.producers
    assert QueueName in client.consumers
    assert len(client.consumers[QueueName]) > 0

    msg = [producer.identity, b"", b"My Message"]
    response = client.send_message(
        message=msg, queue_name=QueueName, worker=consumer.identity
    )

    assert isinstance(response, SyftSuccess)

    sleep(0.5)
    # consumer.receive()
    assert len(received_message) == 1

    msg = [producer.identity, b"", b"My Message"]
    response = client.send_message(message=msg, queue_name="random queue")
    assert isinstance(response, SyftError)

    assert isinstance(client.close(), SyftSuccess)
    sleep(0.5)
    assert client.producers[QueueName].alive is False
    assert client.consumers[QueueName][0].alive is False


def test_zmq_pub_sub(faker: Faker):
    received_messages = []

    pub_port = random.randint(6001, 10004)

    pub_addr = f"tcp://localhost:{pub_port}"

    QueueName = "ABC"

    # Create a producer
    producer = ZMQProducer(
        port=pub_port, queue_name=QueueName, queue_stash=None, context=None
    )

    assert producer.address == pub_addr
    assert isinstance(producer.backend, Socket)
    assert isinstance(producer, ZMQProducer)
    assert producer.queue_name == QueueName
    assert producer.alive

    first_message = faker.sentence().encode()

    class MyMessageHandler(AbstractMessageHandler):
        queue = QueueName

        @staticmethod
        def handle_message(message: bytes):
            received_messages.append(message)

    # Create a consumer
    consumer = ZMQConsumer(
        message_handler=MyMessageHandler,
        address=pub_addr,
        queue_name=QueueName,
    )

    assert isinstance(consumer, ZMQConsumer)
    assert consumer.address == pub_addr
    assert isinstance(consumer.worker, Socket)
    assert consumer.queue_name == QueueName
    assert consumer.alive
    assert consumer.thread is None
    assert consumer.message_handler == MyMessageHandler
    consumer.run()
    sleep(0.2)

    msg = [producer.identity, b"", first_message]
    producer.send(message=msg, worker=consumer.identity)

    # Check if consumer receives the message
    # consumer.receive()
    sleep(0.2)

    # Validate if message was correctly received in the handler
    assert len(received_messages) == 1
    assert first_message in received_messages

    # Close all socket connections
    producer._stop()
    consumer._stop()
    assert producer.alive is False
    assert consumer.alive is False


def test_zmq_queue_manager() -> None:
    config = ZMQQueueConfig()

    assert isinstance(config.client_config, ZMQClientConfig)
    assert config.client_type == ZMQClient

    queue_manager = QueueManager(config=config)

    assert queue_manager.client_config.hostname

    assert isinstance(queue_manager._client, ZMQClient)
    assert len(queue_manager.consumers) == 0
    assert len(queue_manager.producers) == 0

    # Add a Message Handler
    received_messages = []

    QueueName = "my-queue"

    class CustomHandler(AbstractMessageHandler):
        queue_name = QueueName

        @staticmethod
        def handle_message(message: bytes):
            received_messages.append(message)

    producer = queue_manager.create_producer(
        queue_name=QueueName, queue_stash=None, context=None
    )

    assert isinstance(producer, ZMQProducer)

    consumer = queue_manager.create_consumer(
        message_handler=CustomHandler, address=producer.address
    )

    assert isinstance(consumer, ZMQConsumer)

    assert consumer.address == producer.address

    assert len(queue_manager.consumers) == 1
    assert len(queue_manager.producers) == 1
    assert QueueName in queue_manager.consumers
    assert QueueName in queue_manager.producers
    consumer_count = len(queue_manager.consumers[QueueName])
    assert consumer_count == 1

    status = queue_manager.close()
    assert isinstance(status, SyftSuccess)


def test_zmq_client_serde():
    config = ZMQClientConfig()

    client = ZMQClient(config=config)

    bytes_data = syft.serialize(client, to_bytes=True)

    deser = syft.deserialize(bytes_data, from_bytes=True)

    assert type(deser) == type(client)
