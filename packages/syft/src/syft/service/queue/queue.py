# stdlib
from typing import Optional
from typing import Type
from typing import Union

# relative
from ...serde.deserialize import _deserialize as deserialize
from ...serde.serializable import serializable
from ..response import SyftError
from ..response import SyftSuccess
from .base_queue import AbstractMessageHandler
from .base_queue import BaseQueueManager
from .base_queue import QueueConfig
from .queue_stash import QueueItem
from .queue_stash import Status


@serializable()
class QueueManager(BaseQueueManager):
    config: QueueConfig

    def post_init(self):
        self.client_config = self.config.client_config()
        self._client = self.config.client_type(self.client_config)

    def close(self):
        return self._client.close()

    def create_consumer(
        self,
        message_handler: Type[AbstractMessageHandler],
        address: Optional[str] = None,
    ):
        consumer = self._client.add_consumer(
            message_handler=message_handler,
            queue_name=message_handler.queue_name,
            address=address,
        )
        return consumer

    def create_producer(self, queue_name: str):
        return self._client.add_producer(queue_name=queue_name)

    def send(
        self,
        message: bytes,
        queue_name: str,
    ) -> Union[SyftSuccess, SyftError]:
        return self._client.send_message(
            message=message,
            queue_name=queue_name,
        )

    @property
    def producers(self):
        return self._client.producers

    @property
    def consumers(self):
        return self._client.consumers


@serializable()
class APICallMessageHandler(AbstractMessageHandler):
    queue_name = "api_call"

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
