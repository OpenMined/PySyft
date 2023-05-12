# stdlib
from collections import defaultdict
from typing import Type

# relative
from ...serde.deserialize import _deserialize as deserialize
from ...serde.serializable import serializable
from ..response import SyftError
from .base_queue import AbstractMessageHandler
from .base_queue import BaseQueueRouter
from .base_queue import QueueConfig
from .queue_stash import QueueItem
from .queue_stash import Status


class QueueRouter(BaseQueueRouter):
    config: QueueConfig

    def post_init(self):
        self._publisher = None
        self.subscribers = defaultdict(list)
        self.client_config = self.config.client_config()
        self._client = self.config.client_type(self.client_config)

    def start(self):
        self._client.start()

    def close(self):
        for _, subscribers in self.subscribers.items():
            for subscriber in subscribers:
                subscriber.close()
        self.publisher.close()
        self._client.close()

    @property
    def pub_addr(self):
        return self.client_config.pub_addr

    @property
    def sub_addr(self):
        return self.client_config.sub_addr

    def create_subscriber(self, message_handler: Type[AbstractMessageHandler]):
        subscriber = self.config.subscriber(
            message_handler=message_handler,
            address=self.sub_addr,
            queue_name=message_handler.queue,
        )
        self.subscribers[message_handler.queue].append(subscriber)
        return subscriber

    @property
    def publisher(self):
        if self._publisher is None:
            self._publisher = self.config.publisher(self.pub_addr)
        return self._publisher


@serializable()
class APICallMessageHandler(AbstractMessageHandler):
    queue = "api_call"

    @staticmethod
    def handle_message(message: bytes):
        # relative
        from ...node.node import Node

        task_uid, api_call, worker_settings = deserialize(message, from_bytes=True)

        worker = Node(
            id=worker_settings.id,
            name=worker_settings.name,
            signing_key=worker_settings.signing_key,
            document_store_config=worker_settings.document_store_config,
            action_store_config=worker_settings.action_store_config,
            is_subprocess=True,
        )

        item = QueueItem(
            node_uid=worker.id,
            id=task_uid,
            status=Status.PROCESSING,
        )
        worker.queue_stash.set_result(api_call.credentials, item)
        status = Status.COMPLETED

        try:
            result = worker.handle_api_call(api_call)
            if isinstance(result, SyftError):
                status = Status.ERRORED
        except Exception as e:  # nosec
            status = Status.ERRORED
            result = SyftError(message=f"Failed with exception: {e}")

        item = QueueItem(
            node_uid=worker.id,
            id=task_uid,
            result=result,
            resolved=True,
            status=status,
        )

        worker.queue_stash.set_result(worker.verify_key, item)
        worker.queue_stash.partition.close()
