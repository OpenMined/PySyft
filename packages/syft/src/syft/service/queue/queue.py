# stdlib
from typing import Any
from typing import Optional
from typing import Type
from typing import Union

# third party
from result import Err
from result import Ok

# relative
from ...serde.deserialize import _deserialize as deserialize
from ...serde.serializable import serializable
from ...service.context import AuthedServiceContext
from ..job.job_stash import JobStatus
from ..response import SyftError
from ..response import SyftSuccess
from .base_queue import AbstractMessageHandler
from .base_queue import BaseQueueManager
from .base_queue import QueueConfig
from .queue_stash import Status


@serializable()
class QueueManager(BaseQueueManager):
    config: QueueConfig

    def post_init(self):
        self.client_config = self.config.client_config
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

    def create_message_queue(self, queue_name: str):
        return self._client.add_message_queue(queue_name)

    def create_producer(
        self, queue_name: str, queue_stash, context: AuthedServiceContext
    ):
        return self._client.add_producer(
            queue_name=queue_name, queue_stash=queue_stash, context=context
        )

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

        queue_item = deserialize(message, from_bytes=True)
        worker_settings = queue_item.worker_settings

        queue_config = worker_settings.queue_config
        queue_config.client_config.create_producer = False
        queue_config.client_config.n_consumers = 0

        worker = Node(
            id=worker_settings.id,
            name=worker_settings.name,
            signing_key=worker_settings.signing_key,
            document_store_config=worker_settings.document_store_config,
            action_store_config=worker_settings.action_store_config,
            blob_storage_config=worker_settings.blob_store_config,
            queue_config=queue_config,
            is_subprocess=True,
            migrate=False,
        )
        # otherwise it reads it from env, resulting in the wrong credentials
        worker.id = worker_settings.id
        worker.signing_key = worker_settings.signing_key

        credentials = queue_item.syft_client_verify_key

        job_item = worker.job_stash.get_by_uid(credentials, queue_item.job_id).ok()

        queue_item.status = Status.PROCESSING
        queue_item.node_uid = worker.id

        job_item.status = JobStatus.PROCESSING
        job_item.node_uid = worker.id

        worker.queue_stash.set_result(credentials, queue_item)
        worker.job_stash.set_result(credentials, job_item)

        status = Status.COMPLETED
        job_status = JobStatus.COMPLETED

        try:
            call_method = getattr(
                worker.get_service(queue_item.service), queue_item.method
            )

            role = worker.get_role_for_credentials(credentials=credentials)
            context = AuthedServiceContext(
                node=worker,
                credentials=credentials,
                role=role,
                job_id=queue_item.job_id,
                has_execute_permissions=queue_item.has_execute_permissions
            )

            result: Any = call_method(context, *queue_item.args, **queue_item.kwargs)

            if isinstance(result, Ok):
                result = result.ok()
            elif isinstance(result, SyftError) or isinstance(result, Err):
                status = Status.ERRORED
                job_status = JobStatus.ERRORED
        except Exception as e:  # nosec
            status = Status.ERRORED
            job_status = JobStatus.ERRORED
            # stdlib

            raise e
            # result = SyftError(
            #     message=f"Failed with exception: {e}, {traceback.format_exc()}"
            # )
            # print("HAD AN ERROR WHILE HANDLING MESSAGE", result.message)

        queue_item.result = result
        queue_item.resolved = True
        queue_item.status = status

        # get new job item to get latest iter status
        job_item = worker.job_stash.get_by_uid(credentials, job_item.id).ok()

        # if result.is_ok():

        job_item.node_uid = worker.id
        job_item.result = result
        job_item.resolved = True
        job_item.status = job_status

        worker.queue_stash.set_result(credentials, queue_item)
        worker.job_stash.set_result(credentials, job_item)
