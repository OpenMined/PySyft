# stdlib
from collections import defaultdict
from typing import Any

# relative
from .base_queue import AbstractMessageHandler
from .base_queue import BaseQueueRouter
from .base_queue import QueueConfig


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
