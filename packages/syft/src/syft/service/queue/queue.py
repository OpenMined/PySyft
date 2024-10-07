# stdlib
from enum import Enum
import logging
from multiprocessing import Process
import threading
from threading import Thread
import time
from typing import Any
from typing import TYPE_CHECKING
from typing import cast

# third party
import psutil

# relative
from ...serde.deserialize import _deserialize as deserialize
from ...serde.serializable import serializable
from ...server.credentials import SyftVerifyKey
from ...server.worker_settings import WorkerSettings
from ...service.context import AuthedServiceContext
from ...store.linked_obj import LinkedObject
from ...types.datetime import DateTime
from ...types.errors import SyftException
from ...types.uid import UID
from ..job.job_stash import Job
from ..job.job_stash import JobStatus
from ..notification.email_templates import FailedJobTemplate
from ..notification.notification_service import CreateNotification
from ..notifier.notifier_enums import NOTIFIERS
from ..queue.queue_service import QueueService
from ..response import SyftError
from ..response import SyftSuccess
from ..worker.worker_stash import WorkerStash
from .base_queue import AbstractMessageHandler
from .base_queue import BaseQueueManager
from .base_queue import QueueConfig
from .base_queue import QueueConsumer
from .base_queue import QueueProducer
from .queue_stash import QueueItem
from .queue_stash import Status

if TYPE_CHECKING:
    # relative
    from .queue_stash import QueueStash

logger = logging.getLogger(__name__)


@serializable(canonical_name="WorkerType", version=1)
class ConsumerType(str, Enum):
    Thread = "thread"
    Process = "process"
    Synchronous = "synchronous"


class MonitorThread(threading.Thread):
    def __init__(
        self,
        queue_item: QueueItem,
        worker: Any,  # should be of type Worker(Server), but get circular import error
        credentials: SyftVerifyKey,
        interval: int = 5,
    ) -> None:
        super().__init__()
        self.interval = interval
        self.stop_requested = threading.Event()
        self.credentials = credentials
        self.worker = worker
        self.queue_item = queue_item

    def run(self) -> None:
        while not self.stop_requested.is_set():
            self.monitor()
            time.sleep(self.interval)

    def monitor(self) -> None:
        # Implement the monitoring logic here
        job = self.worker.job_stash.get_by_uid(
            self.credentials, self.queue_item.job_id
        ).unwrap()
        if job and job.status == JobStatus.TERMINATING:
            self.terminate(job)
            for subjob in job.subjobs:
                self.terminate(subjob)

            self.queue_item.status = Status.INTERRUPTED
            self.queue_item.resolved = True
            self.worker.queue_stash.set_result(self.credentials, self.queue_item)
            # How about subjobs of subjobs?

    def stop(self) -> None:
        self.stop_requested.set()

    def terminate(self, job: Job) -> None:
        job.resolved = True
        job.status = JobStatus.INTERRUPTED
        self.worker.job_stash.set_result(self.credentials, job)
        try:
            psutil.Process(job.job_pid).terminate()
        except psutil.Error as e:
            logger.warning(f"Failed to terminate job {job.id}: {e}")


@serializable(canonical_name="QueueManager", version=1)
class QueueManager(BaseQueueManager):
    config: QueueConfig

    def post_init(self) -> None:
        self.client_config = self.config.client_config
        self._client = self.config.client_type(self.client_config)

    def close(self) -> SyftSuccess:
        return self._client.close()

    def create_consumer(
        self,
        message_handler: type[AbstractMessageHandler],
        service_name: str,
        worker_stash: WorkerStash | None = None,
        address: str | None = None,
        syft_worker_id: UID | None = None,
    ) -> QueueConsumer:
        consumer = self._client.add_consumer(
            message_handler=message_handler,
            queue_name=message_handler.queue_name,
            address=address,
            service_name=service_name,
            worker_stash=worker_stash,
            syft_worker_id=syft_worker_id,
        )
        return consumer

    def create_producer(
        self,
        queue_name: str,
        queue_stash: "QueueStash",
        context: AuthedServiceContext,
        worker_stash: WorkerStash,
    ) -> QueueProducer:
        return self._client.add_producer(
            queue_name=queue_name,
            queue_stash=queue_stash,
            context=context,
            worker_stash=worker_stash,
        )

    def send(
        self,
        message: bytes,
        queue_name: str,
    ) -> SyftSuccess:
        return self._client.send_message(
            message=message,
            queue_name=queue_name,
        )

    @property
    def producers(self) -> Any:
        return self._client.producers

    @property
    def consumers(self) -> Any:
        return self._client.consumers


def handle_message_multiprocessing(
    worker_settings: WorkerSettings,
    queue_item: QueueItem,
    credentials: SyftVerifyKey,
) -> None:
    # this is a temp hack to prevent some multithreading issues
    time.sleep(0.5)
    queue_config = worker_settings.queue_config
    if queue_config is None:
        raise ValueError(f"{worker_settings} has no queue configurations!")
    queue_config.client_config.create_producer = False
    queue_config.client_config.n_consumers = 0

    # relative
    from ...server.server import Server

    worker = Server(
        id=worker_settings.id,
        name=worker_settings.name,
        signing_key=worker_settings.signing_key,
        db_config=worker_settings.db_config,
        blob_storage_config=worker_settings.blob_store_config,
        server_side_type=worker_settings.server_side_type,
        queue_config=queue_config,
        is_subprocess=True,
        migrate=False,
        deployment_type=worker_settings.deployment_type,
    )

    # otherwise it reads it from env, resulting in the wrong credentials
    worker.id = worker_settings.id
    worker.signing_key = worker_settings.signing_key

    # Set monitor thread for this job.
    monitor_thread = MonitorThread(queue_item, worker, credentials)
    monitor_thread.start()

    if queue_item.service == "user":
        queue_item.service = "userservice"

    # in case of error
    result = None

    try:
        role = worker.get_role_for_credentials(credentials=credentials)

        context = AuthedServiceContext(
            server=worker,
            credentials=credentials,
            role=role,
            job_id=queue_item.job_id,
            has_execute_permissions=queue_item.has_execute_permissions,
        )

        # relative
        from ...server.server import AuthServerContextRegistry

        AuthServerContextRegistry.set_server_context(
            server_uid=worker.id,
            context=context,
            user_verify_key=credentials,
        )

        call_method = getattr(worker.get_service(queue_item.service), queue_item.method)
        result = call_method(context, *queue_item.args, **queue_item.kwargs)
        status = Status.COMPLETED
        job_status = JobStatus.COMPLETED
    except Exception as e:
        root_context = AuthedServiceContext(
            server=context.server,
            credentials=worker.signing_key.verify_key,  # type: ignore
        )
        link = LinkedObject.with_context(
            queue_item, context=root_context, service_type=QueueService
        )
        message = CreateNotification(
            subject=f"Job {queue_item.job_id} failed!",
            from_user_verify_key=worker.signing_key.verify_key,  # type: ignore
            to_user_verify_key=credentials,
            linked_obj=link,
            notifier_types=[NOTIFIERS.EMAIL],
            email_template=FailedJobTemplate,
        )
        method = worker.services.notification.send
        result = method(context=root_context, notification=message)

        status = Status.ERRORED
        job_status = JobStatus.ERRORED
        logger.exception("Unhandled error in handle_message_multiprocessing")
        error_msg = e.public_message if isinstance(e, SyftException) else str(e)
        result = SyftError(message=error_msg)

    queue_item.result = result
    queue_item.resolved = True
    queue_item.status = status

    # get new job item to get latest iter status
    job_item = worker.job_stash.get_by_uid(credentials, queue_item.job_id).unwrap(
        public_message=f"Job {queue_item.job_id} not found!"
    )

    job_item.server_uid = worker.id  # type: ignore[assignment]
    job_item.result = result
    job_item.resolved = True
    job_item.status = job_status

    worker.queue_stash.set_result(credentials, queue_item).unwrap(
        public_message="Failed to set result into QueueItem after running"
    )
    worker.job_stash.set_result(credentials, job_item).unwrap(
        public_message="Failed to set job after running"
    )

    # Finish monitor thread
    monitor_thread.stop()


@serializable(canonical_name="APICallMessageHandler", version=1)
class APICallMessageHandler(AbstractMessageHandler):
    queue_name = "api_call"

    @staticmethod
    def handle_message(message: bytes, syft_worker_id: UID) -> None:
        # relative
        from ...server.server import Server

        queue_item = deserialize(message, from_bytes=True)
        queue_item = cast(QueueItem, queue_item)
        worker_settings = queue_item.worker_settings
        if worker_settings is None:
            raise ValueError("Worker settings are missing in the queue item.")

        queue_config = worker_settings.queue_config
        queue_config.client_config.create_producer = False
        queue_config.client_config.n_consumers = 0

        worker = Server(
            id=worker_settings.id,
            name=worker_settings.name,
            signing_key=worker_settings.signing_key,
            db_config=worker_settings.db_config,
            server_side_type=worker_settings.server_side_type,
            deployment_type=worker_settings.deployment_type,
            queue_config=queue_config,
            is_subprocess=True,
            migrate=False,
        )

        # otherwise it reads it from env, resulting in the wrong credentials
        worker.id = worker_settings.id
        worker.signing_key = worker_settings.signing_key

        credentials = queue_item.syft_client_verify_key
        try:
            job_item: Job = worker.job_stash.get_by_uid(
                credentials, queue_item.job_id
            ).unwrap()  # type: ignore
        except SyftException as exc:
            logger.warning(exc._private_message or exc.public_message)
            raise

        queue_item.status = Status.PROCESSING
        queue_item.server_uid = worker.id

        job_item.status = JobStatus.PROCESSING
        job_item.server_uid = worker.id  # type: ignore[assignment]
        job_item.updated_at = DateTime.now()

        if syft_worker_id is not None:
            job_item.job_worker_id = syft_worker_id

        worker.queue_stash.set_result(credentials, queue_item).unwrap()
        worker.job_stash.set_result(credentials, job_item).unwrap()

        logger.info(
            f"Handling queue item: id={queue_item.id}, method={queue_item.method} "
            f"args={queue_item.args}, kwargs={queue_item.kwargs} "
            f"service={queue_item.service}, as={queue_config.consumer_type}"
        )

        if queue_config.consumer_type == ConsumerType.Thread:
            thread = Thread(
                target=handle_message_multiprocessing,
                args=(worker_settings, queue_item, credentials),
            )
            thread.start()
            thread.join()
        elif queue_config.consumer_type == ConsumerType.Process:
            # if psutil.pid_exists(job_item.job_pid):
            #     psutil.Process(job_item.job_pid).terminate()
            process = Process(
                target=handle_message_multiprocessing,
                args=(worker_settings, queue_item, credentials),
            )
            process.start()
            job_item.job_pid = process.pid
            worker.job_stash.set_result(credentials, job_item).unwrap()
            process.join()
        elif queue_config.consumer_type == ConsumerType.Synchronous:
            handle_message_multiprocessing(worker_settings, queue_item, credentials)
