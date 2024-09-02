# stdlib
from collections import defaultdict
from secrets import token_hex
import sys
from time import sleep

# third party
from faker import Faker
import pytest
from zmq import Socket

# syft absolute
import syft
from syft.service.queue.base_queue import AbstractMessageHandler
from syft.service.queue.queue import QueueManager
from syft.service.queue.zmq_client import ZMQClient
from syft.service.queue.zmq_client import ZMQClientConfig
from syft.service.queue.zmq_client import ZMQQueueConfig
from syft.service.queue.zmq_consumer import ZMQConsumer
from syft.service.queue.zmq_producer import ZMQProducer
from syft.service.response import SyftSuccess
from syft.types.errors import SyftException
from syft.util.util import get_queue_address
from syft.util.util import get_random_available_port


@pytest.fixture
def client():
    hostname = "127.0.0.1"
    config = ZMQClientConfig(hostname=hostname)
    client = ZMQClient(config=config)
    yield client
    # Cleanup code
    client.close()


# @pytest.mark.flaky(reruns=3, reruns_delay=3)
@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
def test_zmq_client(client):
    hostname = "127.0.0.1"

    assert client.config.hostname == hostname

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
        def handle_message(message: bytes, *args, **kwargs):
            received_message.append(message)

    consumer = client.add_consumer(
        queue_name=QueueName,
        message_handler=MyMessageHandler,
        service_name="my-service",
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

    msg = b"My Message"
    response = client.send_message(
        message=msg,
        queue_name=QueueName,
        worker=consumer.id.encode(),
    )

    assert isinstance(response, SyftSuccess)

    sleep(0.5)
    # consumer.receive()
    assert len(received_message) == 1

    msg = b"My Message"
    with pytest.raises(SyftException):
        response = client.send_message(message=msg, queue_name="random queue")

    assert isinstance(client.close(), SyftSuccess)
    sleep(0.5)
    assert client.producers[QueueName].alive is False
    assert client.consumers[QueueName][0].alive is False


@pytest.fixture
def producer():
    pub_port = get_random_available_port()
    QueueName = token_hex(8)

    # Create a producer
    producer = ZMQProducer(
        port=pub_port,
        queue_name=QueueName,
        queue_stash=None,
        worker_stash=None,
        context=None,
    )
    yield producer
    # Cleanup code
    if producer.alive:
        producer.close()
    del producer


@pytest.fixture
def consumer(producer):
    # Create a consumer
    consumer = ZMQConsumer(
        message_handler=None,
        address=producer.address,
        queue_name=producer.queue_name,
        service_name=token_hex(8),
    )
    yield consumer
    # Cleanup code
    if consumer.alive:
        consumer.close()
    del consumer


@pytest.mark.flaky(reruns=3, reruns_delay=3)
@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
def test_zmq_pub_sub(faker: Faker, producer, consumer):
    received_messages = []

    pub_addr = get_queue_address(producer.port)

    assert producer.address == pub_addr
    assert isinstance(producer.socket, Socket)
    assert isinstance(producer, ZMQProducer)
    assert producer.alive

    queue_name = producer.queue_name

    first_message = faker.sentence().encode()

    class MyMessageHandler(AbstractMessageHandler):
        queue = producer.queue_name

        @staticmethod
        def handle_message(message: bytes, *args, **kwargs):
            received_messages.append(message)

    consumer.message_handler = MyMessageHandler

    assert isinstance(consumer, ZMQConsumer)
    assert consumer.address == pub_addr
    assert isinstance(consumer.socket, Socket)
    assert consumer.queue_name == queue_name
    assert consumer.alive
    assert consumer.thread is None
    assert consumer.message_handler == MyMessageHandler
    consumer.run()
    sleep(0.2)

    msg = [producer.id.encode(), b"", first_message]
    producer.send(message=msg, worker=consumer.id.encode())

    # Check if consumer receives the message
    # consumer.receive()
    sleep(0.2)

    # Validate if message was correctly received in the handler
    assert len(received_messages) == 1
    assert first_message in received_messages

    # Close all socket connections
    producer.close()
    consumer.close()
    assert producer.alive is False
    assert consumer.alive is False


@pytest.fixture
def queue_manager():
    # Create a consumer
    config = ZMQQueueConfig()
    queue_manager = QueueManager(config=config)
    yield queue_manager
    # Cleanup code
    queue_manager.close()


@pytest.mark.flaky(reruns=3, reruns_delay=3)
@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
def test_zmq_queue_manager(queue_manager) -> None:
    config = queue_manager.config

    assert isinstance(config.client_config, ZMQClientConfig)
    assert config.client_type == ZMQClient

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
        def handle_message(message: bytes, *args, **kwargs):
            received_messages.append(message)

    producer = queue_manager.create_producer(
        queue_name=QueueName, queue_stash=None, worker_stash=None, context=None
    )

    assert isinstance(producer, ZMQProducer)

    consumer = queue_manager.create_consumer(
        message_handler=CustomHandler,
        address=producer.address,
        service_name="my-service",
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
