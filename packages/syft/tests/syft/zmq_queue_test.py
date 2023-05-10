# stdlib
import random
from typing import Any

# third party
import zmq
from zmq import Socket

# syft absolute
from syft.node.worker_settings import WorkerSettings
from syft.service.queue.zmq_queue import AbstractMessageHandler
from syft.service.queue.zmq_queue import ZMQClient
from syft.service.queue.zmq_queue import ZMQPublisher
from syft.service.queue.zmq_queue import ZMQQueueClientConfig
from syft.service.queue.zmq_queue import ZMQSubscriber


def test_zmq_client():
    pub_port = random.randint(6001, 10004)
    sub_port = random.randint(6001, 10004)

    pub_addr = f"tcp://127.0.0.1:{pub_port}"
    sub_addr = f"tcp://127.0.0.1:{sub_port}"

    config = ZMQQueueClientConfig()
    config.pub_addr = pub_addr
    config.sub_addr = sub_addr

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


def test_zmq_pub_sub(faker, worker):
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
    received_message = received_messages[0]
    assert received_message == second_message

    # Close all socket connections
    publisher.close()
    subscriber.close()
    assert publisher._publisher.closed
    assert subscriber._subscriber.closed
