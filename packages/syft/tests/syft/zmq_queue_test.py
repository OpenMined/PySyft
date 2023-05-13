# stdlib
import random
import time

# third party
from faker import Faker
from zmq import Socket

# syft absolute
import syft
from syft.service.queue.base_queue import AbstractMessageHandler
from syft.service.queue.queue import QueueRouter
from syft.service.queue.zmq_queue import ZMQClient
from syft.service.queue.zmq_queue import ZMQClientConfig
from syft.service.queue.zmq_queue import ZMQConsumer
from syft.service.queue.zmq_queue import ZMQProducer
from syft.service.queue.zmq_queue import ZMQQueueConfig


def test_zmq_client():
    hostname = "127.0.0.1"

    config = ZMQClientConfig(hostname=hostname)

    assert config.hostname == hostname

    client = ZMQClient(config=config)

    assert client.host == hostname
    assert len(client.producers) == 0
    assert len(client.consumers) == 0

    client.producers

    client.close()


def test_zmq_pub_sub(faker: Faker):
    received_messages = []

    pub_port = random.randint(6001, 10004)

    pub_addr = f"tcp://127.0.0.1:{pub_port}"

    # Create a publisher
    publisher = ZMQProducer(address=pub_addr)
    queue_name = "ABC"

    assert publisher.address == pub_addr
    assert isinstance(publisher._producer, Socket)

    first_message = faker.sentence().encode()

    # The first message will be dropped since no subscriber
    # queues are attached yet.
    publisher.send(first_message, queue_name=queue_name)

    class MyMessageHandler(AbstractMessageHandler):
        queue = queue_name

        @staticmethod
        def handle_message(message: bytes):
            received_messages.append(message)

    # Create a queue and subscriber
    subscriber = ZMQConsumer(
        message_handler=MyMessageHandler,
        address=pub_addr,
        queue_name=queue_name,
    )

    # Add sleep for subscriber to connect in green thread
    time.sleep(1)

    assert subscriber.address == pub_addr
    assert isinstance(subscriber._consumer, Socket)
    assert subscriber.recv_thread is None

    # Send in a second message
    second_message = faker.sentence().encode()

    publisher.send(message=second_message, queue_name=queue_name)

    # Check if subscriber receives the message
    received_message = subscriber._consumer.recv_multipart()

    # Validate contents of the message
    assert len(received_message) == 2
    assert received_message[0] == queue_name.encode()
    assert received_message[1] == second_message

    # Send in another message
    publisher.send(message=second_message, queue_name=queue_name)

    # Receive message via the message handler
    subscriber.receive()

    # Validate if message was correctly received in the handler
    assert len(received_messages) == 1
    received_message = received_messages[0]
    assert received_message == second_message

    # Close all socket connections
    publisher.close()
    subscriber.close()
    assert publisher._producer.closed
    assert subscriber._consumer.closed


def test_zmq_queue_router() -> None:
    config = ZMQQueueConfig()

    assert config.client_config == ZMQClientConfig
    assert config.client_type == ZMQClient
    assert config.publisher == ZMQProducer
    assert config.subscriber == ZMQConsumer

    queue_router = QueueRouter(config=config)

    assert queue_router.client_config.pub_addr
    assert queue_router.client_config.sub_addr

    assert len(queue_router.subscribers) == 0
    assert isinstance(queue_router._client, ZMQClient)
    assert queue_router._publisher is None

    # start the queue_router
    queue_router.start()
    assert queue_router._client.thread

    assert isinstance(queue_router.publisher, ZMQProducer)

    # Add a Message Handler
    received_messages = []

    queue_name = "my-queue"

    class CustomHandler(AbstractMessageHandler):
        queue = queue_name

        @staticmethod
        def handle_message(message: bytes):
            received_messages.append(message)

    subscriber = queue_router.create_consumer(
        message_handler=CustomHandler,
    )

    assert isinstance(subscriber, ZMQConsumer)

    assert len(queue_router.subscribers) == 1
    assert queue_name in queue_router.subscribers.keys()
    assert queue_router.subscribers[queue_name]
    subscriber_count = len(queue_router.subscribers[queue_name])
    assert subscriber_count == 1
    assert queue_router.subscribers[queue_name][0] == subscriber

    queue_router.close()


def test_zmq_client_serde():
    config = ZMQClientConfig()

    client = ZMQClient(config=config)

    bytes_data = syft.serialize(client, to_bytes=True)

    deser = syft.deserialize(bytes_data, from_bytes=True)

    assert type(deser) == type(client)
