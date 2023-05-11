# stdlib
from collections import defaultdict
from typing import Any

# relative
from ...serde.deserialize import _deserialize as deserialize
from .base_queue import AbstractMessageHandler
from .base_queue import BaseQueueRouter
from .base_queue import QueueConfig
from .queue_stash import QueueItem


class QueueRouter(BaseQueueRouter):
    config: QueueConfig

    def post_init(self):
        self._publisher = None
        self.subscribers = defaultdict(list)
        self._client = self.config.client_type(self.config.client_config)

    def start(self):
        self._client.start()

    def close(self):
        self._client.close()

    @property
    def pub_addr(self):
        return self.config.client_config.pub_addr

    @property
    def sub_addr(self):
        return self.config.client_config.sub_addr

    def create_subscriber(
        self, message_handler: AbstractMessageHandler, worker_settings: Any
    ):
        subscriber = self.config.subscriber(
            message_handler=message_handler.message_handler,
            address=self.sub_addr,
            worker_settings=worker_settings,
            queue_name=message_handler.queue,
        )
        self.subscribers[message_handler.queue].append(subscriber)
        return subscriber

    @property
    def publisher(self):
        if self._publisher is None:
            self._publisher = self.config.publisher(self.pub_addr)
        return self._publisher


class APICallMessageHandler(AbstractMessageHandler):
    queue = "api_call"

    @classmethod
    def message_handler(cls, message: bytes, worker: Any):
        task_uid, api_call = deserialize(message, from_bytes=True)
        result = worker.handle_api_call(api_call)
        item = QueueItem(
            node_uid=worker.id,
            id=task_uid,
            result=result,
            resolved=True,
        )
        worker.queue_stash.set_result(api_call.credentials, item)
        worker.queue_stash.partition.close()
