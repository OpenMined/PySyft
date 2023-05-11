# stdlib
import random
from typing import Any

# third party
from faker import Faker
import zmq
from zmq import Socket

# syft absolute
from syft.node.node import Node
from syft.node.worker_settings import WorkerSettings
from syft.service.queue.base_queue import AbstractMessageHandler
from syft.service.queue.queue import QueueRouter
from syft.service.queue.zmq_queue import ZMQClient
from syft.service.queue.zmq_queue import ZMQClientConfig
from syft.service.queue.zmq_queue import ZMQPublisher
from syft.service.queue.zmq_queue import ZMQQueueConfig
from syft.service.queue.zmq_queue import ZMQSubscriber


def test_zmq_client():
    pub_port = random.randint(6001, 10004)
    sub_port = random.randint(6001, 10004)

    pub_addr = f"tcp://127.0.0.1:{pub_port}"
    sub_addr = f"tcp://127.0.0.1:{sub_port}"

    config = ZMQClientConfig(pub_addr=pub_addr, sub_addr=sub_addr)

    client = ZMQClient(config=config)

    assert client.sub_addr == sub_addr
    assert client.pub_addr == pub_addr
    assert client.context is not None
    assert client.logger_thread is None
    assert client.thread is None

    client.start()

    assert client.thread is not None
    assert client.thread.dead is False

    assert client.logger_thread is not None
    assert client.logger_thread.dead is False

    assert isinstance(client.xsub, Socket)
    assert isinstance(client.xpub, Socket)

    assert client.xpub.socket_type == zmq.XPUB
    assert client.xsub.socket_type == zmq.XSUB

    assert client.mon_addr is not None
    assert isinstance(client.mon_pub, Socket)
    assert isinstance(client.mon_sub, Socket)

    assert client.mon_pub.socket_type == zmq.PAIR
    assert client.mon_sub.socket_type == zmq.PAIR

    client.close()

    assert client.context
    assert client.thread.dead
    assert client.logger_thread.dead


def test_zmq_pub_sub(faker: Faker, worker: Node):
    received_messages = []

    class MyMessageHandler(AbstractMessageHandler):
        @classmethod
        def message_handler(cls, message: bytes, worker: Any):
            received_messages.append(message)

    pub_port = random.randint(6001, 10004)

    pub_addr = f"tcp://127.0.0.1:{pub_port}"

    # Create a publisher
    publisher = ZMQPublisher(address=pub_addr)
    queue_name = "ABC"

    worker_settings = WorkerSettings.from_node(worker)

    assert publisher.address == pub_addr
    assert isinstance(publisher._publisher, Socket)

    first_message = faker.sentence().encode()

    # The first message will be dropped since no subscriber
    # queues are attached yet.
    publisher.send(first_message, queue_name=queue_name)

    # Create a queue and subscriber
    subscriber = ZMQSubscriber(
        worker_settings=worker_settings,
        message_handler=MyMessageHandler.message_handler,
        address=pub_addr,
        queue_name=queue_name,
    )

    assert subscriber.address == pub_addr
    assert isinstance(subscriber._subscriber, Socket)

    # Send in a second message
    second_message = faker.sentence().encode()

    publisher.send(message=second_message, queue_name=queue_name)

    # Check if subscriber receives the message
    received_message = subscriber._subscriber.recv_multipart()

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
    assert publisher._publisher.closed
    assert subscriber._subscriber.closed


def test_zmq_queue_router(worker: Node) -> None:
    pub_port = random.randint(6001, 10004)
    sub_port = random.randint(6001, 10004)

    pub_addr = f"tcp://127.0.0.1:{pub_port}"
    sub_addr = f"tcp://127.0.0.1:{sub_port}"

    config = ZMQQueueConfig()

    assert isinstance(config.client_config, ZMQClientConfig)
    assert config.client_type == ZMQClient
    assert config.publisher == ZMQPublisher
    assert config.subscriber == ZMQSubscriber

    config.client_config.pub_addr = pub_addr
    config.client_config.sub_addr = sub_addr

    queue_router = QueueRouter(config=config)

    assert queue_router.pub_addr == pub_addr
    assert queue_router.sub_addr == sub_addr
    assert len(queue_router.subscribers) == 0
    assert isinstance(queue_router._client, ZMQClient)
    assert queue_router._publisher is None

    # start the queue_router
    queue_router.start()
    assert queue_router._client.thread

    assert isinstance(queue_router.publisher, ZMQPublisher)

    # Add a Message Handler
    received_messages = []

    queue_name = "my-queue"

    class CustomHandler(AbstractMessageHandler):
        queue = queue_name

        @classmethod
        def message_handler(cls, message: bytes, worker: Any):
            received_messages.append(message)

    worker_settings = WorkerSettings.from_node(worker)

    subscriber = queue_router.create_subscriber(
        message_handler=CustomHandler,
        worker_settings=worker_settings,
    )

    assert isinstance(subscriber, ZMQSubscriber)

    assert len(queue_router.subscribers) == 1
    assert queue_name in queue_router.subscribers.keys()
    assert queue_router.subscribers[queue_name]
    subscriber_count = len(queue_router.subscribers[queue_name])
    assert subscriber_count == 1
    assert queue_router.subscribers[queue_name][0] == subscriber

    queue_router.close()
