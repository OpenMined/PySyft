# stdlib
from collections.abc import Callable
import typing

# relative
from ..service.action.action_service import ActionService
from ..service.action.action_store import ActionStore
from ..service.api.api_service import APIService
from ..service.attestation.attestation_service import AttestationService
from ..service.blob_storage.service import BlobStorageService
from ..service.code.status_service import UserCodeStatusService
from ..service.code.user_code_service import UserCodeService
from ..service.code_history.code_history_service import CodeHistoryService
from ..service.data_subject.data_subject_member_service import DataSubjectMemberService
from ..service.data_subject.data_subject_service import DataSubjectService
from ..service.dataset.dataset_service import DatasetService
from ..service.enclave.enclave_service import EnclaveService
from ..service.job.job_service import JobService
from ..service.log.log_service import LogService
from ..service.metadata.metadata_service import MetadataService
from ..service.migration.migration_service import MigrationService
from ..service.network.network_service import NetworkService
from ..service.notification.notification_service import NotificationService
from ..service.notifier.notifier_service import NotifierService
from ..service.output.output_service import OutputService
from ..service.policy.policy_service import PolicyService
from ..service.project.project_service import ProjectService
from ..service.queue.queue_service import QueueService
from ..service.request.request_service import RequestService
from ..service.service import AbstractService
from ..service.settings.settings_service import SettingsService
from ..service.sync.sync_service import SyncService
from ..service.user.user_service import UserService
from ..service.worker.image_registry_service import SyftImageRegistryService
from ..service.worker.worker_image_service import SyftWorkerImageService
from ..service.worker.worker_pool_service import SyftWorkerPoolService
from ..service.worker.worker_service import WorkerService
from .node import Node


class ServiceRegistry:
    # Services
    action: ActionService
    user: UserService
    attestation: AttestationService
    worker: WorkerService
    settings: SettingsService
    dataset: DatasetService
    user_code: UserCodeService
    log: LogService
    request: RequestService
    queue: QueueService
    job: JobService
    api: APIService
    data_subject: DataSubjectService
    network: NetworkService
    policy: PolicyService
    notifier: NotifierService
    notification: NotificationService
    data_subject_member: DataSubjectMemberService
    project: ProjectService
    enclave: EnclaveService
    code_history: CodeHistoryService
    metadata: MetadataService
    blob_storage: BlobStorageService
    migration: MigrationService
    syft_worker_image: SyftWorkerImageService
    syft_worker_pool: SyftWorkerPoolService
    syft_image_registry: SyftImageRegistryService
    sync: SyncService
    output: OutputService
    user_code_status: UserCodeStatusService

    def __init__(self, node: Node) -> None:
        self.node = node
        self.service_classes = self.get_service_classes()
        self.services: list[AbstractService] = []
        self.service_path_map: dict[str, AbstractService] = {}
        self._construct_services()

    @classmethod
    def get_service_classes(
        cls,
    ) -> dict[str, type[AbstractService]]:
        return {
            name: cls
            for name, cls in typing.get_type_hints(cls).items()
            if issubclass(cls, AbstractService)
        }

    def _construct_services(self) -> None:
        for field_name, service_cls in self.get_service_classes().items():
            if issubclass(service_cls.store_type, ActionStore):
                svc_kwargs = {"store": self.node.action_store}
            else:
                svc_kwargs = {"store": self.node.document_store}

            service = service_cls(**svc_kwargs)
            setattr(self, field_name, service)
            self.services.append(service)
            self.service_path_map[service.__class__.__name__.lower()] = service

    def get_service(self, path_or_func: str | Callable) -> AbstractService:
        if callable(path_or_func):
            path_or_func = path_or_func.__qualname__
        return self._get_service_from_path(path_or_func)

    def _get_service_from_path(self, path: str) -> AbstractService:
        try:
            path_list = path.split(".")
            if len(path_list) > 1:
                _ = path_list.pop()
            service_name = path_list.pop()
            return self.service_path_map[service_name.lower()]
        except KeyError:
            raise ValueError(f"Service {path} not found.")
