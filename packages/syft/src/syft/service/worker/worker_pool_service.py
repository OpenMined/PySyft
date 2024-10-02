# stdlib
import logging
from typing import Any

# third party
import pydantic

# relative
from ...custom_worker.config import DockerWorkerConfig
from ...custom_worker.config import PrebuiltWorkerConfig
from ...custom_worker.config import WorkerConfig
from ...custom_worker.k8s import IN_KUBERNETES
from ...custom_worker.runner_k8s import KubernetesRunner
from ...serde.serializable import serializable
from ...store.db.db import DBManager
from ...store.document_store_errors import NotFoundException
from ...store.document_store_errors import StashException
from ...store.linked_obj import LinkedObject
from ...types.dicttuple import DictTuple
from ...types.errors import SyftException
from ...types.result import as_result
from ...types.uid import UID
from ..context import AuthedServiceContext
from ..request.request import Change
from ..request.request import CreateCustomImageChange
from ..request.request import CreateCustomWorkerPoolChange
from ..request.request import Request
from ..request.request import SubmitRequest
from ..response import SyftSuccess
from ..service import AbstractService
from ..service import SERVICE_TO_TYPES
from ..service import TYPE_TO_SERVICE
from ..service import service_method
from ..user.user_roles import DATA_OWNER_ROLE_LEVEL
from ..user.user_roles import DATA_SCIENTIST_ROLE_LEVEL
from .image_identifier import SyftWorkerImageIdentifier
from .utils import DEFAULT_WORKER_POOL_NAME
from .utils import get_orchestration_type
from .utils import run_containers
from .utils import run_workers_in_threads
from .utils import scale_kubernetes_pool
from .worker_image import SyftWorkerImage
from .worker_image_stash import SyftWorkerImageStash
from .worker_pool import ContainerSpawnStatus
from .worker_pool import WorkerPool
from .worker_pool_stash import SyftWorkerPoolStash
from .worker_service import WorkerService
from .worker_stash import WorkerStash

logger = logging.getLogger(__name__)


@serializable(canonical_name="SyftWorkerPoolService", version=1)
class SyftWorkerPoolService(AbstractService):
    stash: SyftWorkerPoolStash

    def __init__(self, store: DBManager) -> None:
        self.stash = SyftWorkerPoolStash(store=store)
        self.image_stash = SyftWorkerImageStash(store=store)

    @as_result(StashException)
    def pool_exists(self, context: AuthedServiceContext, pool_name: str) -> bool:
        try:
            self.stash.get_by_name(context.credentials, pool_name=pool_name).unwrap()
            return True
        except NotFoundException:
            return False

    @as_result(StashException)
    def image_exists(self, context: AuthedServiceContext, uid: UID) -> bool:
        try:
            self.image_stash.get_by_uid(context.credentials, uid=uid).unwrap()
            return True
        except NotFoundException:
            return False

    @service_method(
        path="worker_pool.launch",
        name="launch",
        roles=DATA_OWNER_ROLE_LEVEL,
    )
    def launch(
        self,
        context: AuthedServiceContext,
        pool_name: str,
        image_uid: UID | None,
        num_workers: int,
        registry_username: str | None = None,
        registry_password: str | None = None,
        pod_annotations: dict[str, str] | None = None,
        pod_labels: dict[str, str] | None = None,
    ) -> list[ContainerSpawnStatus]:
        """Creates a pool of workers from the given SyftWorkerImage.

        - Retrieves the image for the given UID
        - Use docker to launch containers for given image
        - For each successful container instantiation create a SyftWorker object
        - Creates a SyftWorkerPool object

        Args:
            context (AuthedServiceContext): context passed to the service
            name (str): name of the pool
            image_id (UID): UID of the SyftWorkerImage against which the pool should be created
            num_workers (int): the number of SyftWorker that needs to be created in the pool
        """

        pool_exists = self.pool_exists(context, pool_name=pool_name).unwrap()
        if pool_exists:
            raise SyftException(
                public_message=f"Worker Pool with name: {pool_name} already exists !!"
            )

        # If image uid is not passed, then use the default worker image
        # to create the worker pool
        if image_uid is None:
            default_worker_pool = self.stash.get_by_name(
                context.credentials, pool_name=DEFAULT_WORKER_POOL_NAME
            ).unwrap()
            image_uid = default_worker_pool.image_id

        # Get the image object for the given image id
        worker_image = self.image_stash.get_by_uid(
            credentials=context.credentials, uid=image_uid
        ).unwrap()

        worker_stash = context.server.services.worker.stash

        # Create worker pool from given image, with the given worker pool
        # and with the desired number of workers
        worker_list, container_statuses = _create_workers_in_pool(
            context=context,
            pool_name=pool_name,
            existing_worker_cnt=0,
            worker_cnt=num_workers,
            worker_image=worker_image,
            worker_stash=worker_stash,
            registry_username=registry_username,
            registry_password=registry_password,
            pod_annotations=pod_annotations,
            pod_labels=pod_labels,
        ).unwrap()

        # Update the Database with the pool information
        worker_pool = WorkerPool(
            name=pool_name,
            max_count=num_workers,
            image_id=worker_image.id,
            worker_list=worker_list,
            syft_server_location=context.server.id,
            syft_client_verify_key=context.credentials,
        )
        self.stash.set(credentials=context.credentials, obj=worker_pool).unwrap()
        return container_statuses

    @service_method(
        path="worker_pool.create_pool_request",
        name="pool_creation_request",
        roles=DATA_SCIENTIST_ROLE_LEVEL,
    )
    def create_pool_request(
        self,
        context: AuthedServiceContext,
        pool_name: str,
        num_workers: int,
        image_uid: UID,
        reason: str | None = "",
        pod_annotations: dict[str, str] | None = None,
        pod_labels: dict[str, str] | None = None,
    ) -> Request:
        """
        Create a request to launch the worker pool based on a built image.

        Args:
            context (AuthedServiceContext): The authenticated service context.
            pool_name (str): The name of the worker pool.
            num_workers (int): The number of workers in the pool.
            image_uid (Optional[UID]): The UID of the built image.
            reason (Optional[str], optional): The reason for creating the
                worker pool. Defaults to "".
        """
        # Check if image exists for the given image id
        worker_image_exists = self.image_exists(context, uid=image_uid).unwrap()

        # Raise error if worker image doesn't exists
        if not worker_image_exists:
            raise SyftException(
                public_message=f"No image exists for given image uid : {image_uid}"
            )

        # Check if pool already exists for the given pool name
        worker_pool_exists = self.pool_exists(context, pool_name=pool_name).unwrap()

        if worker_pool_exists:
            raise SyftException(
                public_message=f"Worker pool already exists for given pool name: {pool_name}"
            )

        # If no worker pool exists for given pool name
        # and image exists for given image uid, then create a change
        # request object to create the pool with the desired number of workers
        create_worker_pool_change = CreateCustomWorkerPoolChange(
            pool_name=pool_name,
            num_workers=num_workers,
            image_uid=image_uid,
            pod_annotations=pod_annotations,
            pod_labels=pod_labels,
        )
        changes: list[Change] = [create_worker_pool_change]

        # Create a the request object with the changes and submit it
        # for approval.
        request = SubmitRequest(changes=changes)
        return context.server.services.request.submit(
            context=context, request=request, reason=reason
        )

    @service_method(
        path="worker_pool.create_image_and_pool_request",
        name="create_image_and_pool_request",
        roles=DATA_SCIENTIST_ROLE_LEVEL,
    )
    def create_image_and_pool_request(
        self,
        context: AuthedServiceContext,
        pool_name: str,
        num_workers: int,
        config: WorkerConfig,
        tag: str | None = None,
        registry_uid: UID | None = None,
        reason: str | None = "",
        pull_image: bool = True,
        pod_annotations: dict[str, str] | None = None,
        pod_labels: dict[str, str] | None = None,
    ) -> Request:
        """
        Create a request to launch the worker pool based on a built image.

        Args:
            context (AuthedServiceContext): The authenticated service context.
            pool_name (str): The name of the worker pool.
            num_workers (int): The number of workers in the pool.
            config: (WorkerConfig): Config of the image to be built.
            tag (str | None, optional):
                a human-readable manifest identifier that is typically a specific version or variant of an image,
                only needed for `DockerWorkerConfig` to tag the image after it is built.
            reason (str | None, optional): The reason for creating the worker image and pool. Defaults to "".
        """
        if not isinstance(config, DockerWorkerConfig | PrebuiltWorkerConfig):
            raise SyftException(
                public_message="We only support either `DockerWorkerConfig` or `PrebuiltWorkerConfig`."
            )

        if isinstance(config, DockerWorkerConfig):
            if tag is None:
                raise SyftException(
                    public_message="`tag` is required for `DockerWorkerConfig`."
                )

            # Validate image tag
            try:
                SyftWorkerImageIdentifier.from_str(tag=tag)
            except pydantic.ValidationError as e:
                raise SyftException(public_message=f"Invalid `tag`: {e}.")

            if IN_KUBERNETES and registry_uid is None:
                raise SyftException(
                    public_message="`registry_uid` is required in Kubernetes mode for `DockerWorkerConfig`."
                )

        # Check if an image already exists for given docker config
        worker_image_exists = self.image_stash.worker_config_exists(
            credentials=context.credentials, config=config
        ).unwrap()

        if worker_image_exists:
            raise SyftException(
                public_message="Image already exists for given config. \
                    Please use `worker_pool.create_pool_request` to request pool creation."
            )

        # create a list of Change objects and submit a
        # request for these changes for approval
        changes: list[Change] = []

        # Add create custom image change
        # If this change is approved, then build an image using the config
        create_custom_image_change = CreateCustomImageChange(
            config=config,
            tag=tag,
            registry_uid=registry_uid,
            pull_image=pull_image,
        )

        # Check if a pool already exists for given pool name
        worker_pool_exists = self.pool_exists(context, pool_name=pool_name).unwrap()

        # Raise an error if worker pool already exists for the given worker pool name
        if worker_pool_exists:
            raise SyftException(
                public_message=f"Worker Pool with name: {pool_name} already"
                f" exists. Please choose another name!"
            )

        # Add create worker pool change
        # If change is approved then worker pool is created and
        # the desired number of workers are added to the pool
        create_worker_pool_change = CreateCustomWorkerPoolChange(
            pool_name=pool_name,
            num_workers=num_workers,
            config=config,
            pod_annotations=pod_annotations,
            pod_labels=pod_labels,
        )
        changes += [create_custom_image_change, create_worker_pool_change]

        # Create a request object and submit a request for approval
        request = SubmitRequest(changes=changes)
        return context.server.services.request.submit(
            context=context, request=request, reason=reason
        )

    @service_method(
        path="worker_pool.get_all",
        name="get_all",
        roles=DATA_SCIENTIST_ROLE_LEVEL,
    )
    def get_all(self, context: AuthedServiceContext) -> DictTuple[str, WorkerPool]:
        # TODO: During get_all, we should dynamically make a call to docker to get the status of the containers
        # and update the status of the workers in the pool.
        worker_pools = self.stash.get_all(credentials=context.credentials).unwrap()

        res = ((pool.name, pool) for pool in worker_pools)
        return DictTuple(res)

    @service_method(
        path="worker_pool.add_workers",
        name="add_workers",
        roles=DATA_OWNER_ROLE_LEVEL,
    )
    def add_workers(
        self,
        context: AuthedServiceContext,
        number: int,
        pool_id: UID | None = None,
        pool_name: str | None = None,
        registry_username: str | None = None,
        registry_password: str | None = None,
    ) -> list[ContainerSpawnStatus]:
        """Add workers to existing worker pool.

        Worker pool is fetched either using the unique pool id or pool name.

        Args:
            context (AuthedServiceContext): _description_
            number (int): number of workers to add
            pool_id (Optional[UID], optional): Unique UID of the pool. Defaults to None.
            pool_name (Optional[str], optional): Unique name of the pool. Defaults to None.

        Returns:
            List[ContainerSpawnStatus]: List of spawned workers with their status and error if any.
        """

        if number <= 0:
            raise SyftException(public_message=f"Invalid number of workers: {number}")

        # Extract pool using either using pool id or pool name
        if pool_id:
            worker_pool = self.stash.get_by_uid(
                credentials=context.credentials, uid=pool_id
            ).unwrap()
        elif pool_name:
            worker_pool = self.stash.get_by_name(
                credentials=context.credentials,
                pool_name=pool_name,
            ).unwrap()

        existing_worker_cnt = len(worker_pool.worker_list)

        worker_image = self.image_stash.get_by_uid(
            credentials=context.credentials,
            uid=worker_pool.image_id,
        ).unwrap()

        worker_stash = context.server.services.worker.stash

        # Add workers to given pool from the given image
        worker_list, container_statuses = _create_workers_in_pool(
            context=context,
            pool_name=worker_pool.name,
            existing_worker_cnt=existing_worker_cnt,
            worker_cnt=number,
            worker_image=worker_image,
            worker_stash=worker_stash,
            registry_username=registry_username,
            registry_password=registry_password,
        ).unwrap()

        worker_pool.worker_list += worker_list
        worker_pool.max_count = existing_worker_cnt + number

        self.stash.update(credentials=context.credentials, obj=worker_pool).unwrap()
        return container_statuses

    @service_method(
        path="worker_pool.scale",
        name="scale",
        roles=DATA_OWNER_ROLE_LEVEL,
        unwrap_on_success=False,
    )
    def scale(
        self,
        context: AuthedServiceContext,
        number: int,
        pool_id: UID | None = None,
        pool_name: str | None = None,
    ) -> SyftSuccess:
        """
        Scale the worker pool to the given number of workers in Kubernetes.
        Allows both scaling up and down the worker pool.
        """

        client_warning = ""

        if not IN_KUBERNETES:
            raise SyftException(
                public_message="Scaling is only supported in Kubernetes mode"
            )
        elif number < 0:
            # zero is a valid scale down
            raise SyftException(public_message=f"Invalid number of workers: {number}")

        worker_pool: Any = self._get_worker_pool(context, pool_id, pool_name).unwrap()
        current_worker_count = len(worker_pool.worker_list)

        if current_worker_count == number:
            return SyftSuccess(message=f"Worker pool already has {number} workers")
        elif number > current_worker_count:
            workers_to_add = number - current_worker_count
            self.add_workers(
                context=context,
                number=workers_to_add,
                pool_id=pool_id,
                pool_name=pool_name,
                # kube scaling doesn't require password as it replicates an existing deployment
                registry_username=None,
                registry_password=None,
            )
        else:
            # scale down at kubernetes control plane
            runner = KubernetesRunner()
            scale_kubernetes_pool(
                runner,
                pool_name=worker_pool.name,
                replicas=number,
            ).unwrap()

            # scale down removes the last "n" workers
            # workers to delete = len(workers) - number
            workers_to_delete = worker_pool.worker_list[
                -(current_worker_count - number) :
            ]

            worker_stash = context.server.services.worker.stash
            # delete linkedobj workers
            for worker in workers_to_delete:
                worker_stash.delete_by_uid(
                    credentials=context.credentials,
                    uid=worker.object_uid,
                ).unwrap()

            client_warning += "Scaling down workers doesn't kill the associated jobs. Please delete them manually."

            # update worker_pool
            worker_pool.max_count = number
            worker_pool.worker_list = worker_pool.worker_list[:number]
            self.stash.update(
                credentials=context.credentials,
                obj=worker_pool,
            ).unwrap(
                public_message=(
                    f"Pool {worker_pool.name} was scaled down, "
                    f"but failed to update the stash"
                )
            )

        return SyftSuccess(
            message=f"Worker pool scaled to {number} workers",
            client_warnings=[client_warning] if client_warning else [],
        )

    @service_method(
        path="worker_pool.filter_by_image_id",
        name="filter_by_image_id",
        roles=DATA_SCIENTIST_ROLE_LEVEL,
    )
    def filter_by_image_id(
        self, context: AuthedServiceContext, image_uid: UID
    ) -> list[WorkerPool]:
        return self.stash.get_by_image_uid(context.credentials, image_uid).unwrap()

    @service_method(
        path="worker_pool.get_by_name",
        name="get_by_name",
        roles=DATA_SCIENTIST_ROLE_LEVEL,
    )
    def get_by_name(
        self, context: AuthedServiceContext, pool_name: str
    ) -> list[WorkerPool]:
        return self.stash.get_by_name(context.credentials, pool_name).unwrap()

    @service_method(
        path="worker_pool.sync_pool_from_request",
        name="sync_pool_from_request",
        roles=DATA_SCIENTIST_ROLE_LEVEL,
        unwrap_on_success=False,
    )
    def sync_pool_from_request(
        self,
        context: AuthedServiceContext,
        request: Request,
    ) -> Request:
        """Re-submit request from a different server"""

        num_of_changes = len(request.changes)
        pool_name, num_workers, config, image_uid, tag = None, None, None, None, None

        if num_of_changes > 2:
            raise SyftException(
                public_message=f"Invalid pool request object. Only pool request changes allowed. {request.changes}"
            )

        for change in request.changes:
            if isinstance(change, CreateCustomWorkerPoolChange):
                pool_name = change.pool_name
                num_workers = change.num_workers
                image_uid = change.image_uid
                pod_annotations = change.pod_annotations
                pod_labels = change.pod_labels
            elif isinstance(change, CreateCustomImageChange):  # type: ignore[unreachable]
                config = change.config
                tag = change.tag

        if config is None and image_uid is not None:
            return self.create_pool_request(
                context=context,
                pool_name=pool_name,
                num_workers=num_workers,
                image_uid=image_uid,
                pod_annotations=pod_annotations,
                pod_labels=pod_labels,
            )
        elif config is not None:
            return self.create_image_and_pool_request(  # type: ignore[unreachable]
                context=context,
                pool_name=pool_name,
                num_workers=num_workers,
                config=config,
                tag=tag,
                pod_annotations=pod_annotations,
                pod_labels=pod_labels,
            )
        else:
            raise SyftException(
                public_message=(
                    f"Invalid request object: invalid image uid or config in the request changes: "
                    f"{request.changes}"
                )
            )

    @service_method(
        path="worker_pool.delete",
        name="delete",
        roles=DATA_OWNER_ROLE_LEVEL,
        unwrap_on_success=False,
    )
    def delete(
        self,
        context: AuthedServiceContext,
        pool_id: UID | None = None,
        pool_name: str | None = None,
    ) -> SyftSuccess:
        worker_pool = self._get_worker_pool(
            context, pool_id=pool_id, pool_name=pool_name
        ).unwrap(public_message=f"Failed to get WorkerPool: {pool_id or pool_name}")

        uid = worker_pool.id

        self.purge_workers(context=context, pool_id=pool_id, pool_name=pool_name)

        self.stash.delete_by_uid(credentials=context.credentials, uid=uid).unwrap(
            public_message=f"Failed to delete WorkerPool: {worker_pool.name} from stash"
        )

        return SyftSuccess(message=f"Successfully deleted worker pool with id {uid}")

    @service_method(
        path="worker_pool.purge_workers",
        name="purge_workers",
        roles=DATA_OWNER_ROLE_LEVEL,
        unwrap_on_success=False,
    )
    def purge_workers(
        self,
        context: AuthedServiceContext,
        pool_id: UID | None = None,
        pool_name: str | None = None,
    ) -> SyftSuccess:
        worker_pool = self._get_worker_pool(
            context, pool_id=pool_id, pool_name=pool_name
        ).unwrap(public_message=f"Failed to get WorkerPool: {pool_id or pool_name}")

        uid = worker_pool.id

        # relative
        from ..queue.queue_stash import Status

        queue_items = context.server.services.queue.stash._get_by_worker_pool(
            credentials=context.credentials,
            worker_pool=LinkedObject.from_obj(
                obj=worker_pool,
                service_type=self.__class__,
                server_uid=context.server.id,
            ),
        ).unwrap(
            public_message=f"Failed to get queue items mapped to WorkerPool: {worker_pool.name}"
        )

        items_to_interrupt = (
            item
            for item in queue_items
            if item.status in (Status.CREATED, Status.PROCESSING)
        )

        for item in items_to_interrupt:
            item.status = Status.INTERRUPTED
            context.server.services.queue.stash.update(
                credentials=context.credentials,
                obj=item,
            ).unwrap()

        if IN_KUBERNETES:
            # Scale the workers to zero
            runner = KubernetesRunner()
            if runner.exists(worker_pool.name):
                self.scale(context=context, number=0, pool_id=uid)
                runner.delete_pool(pool_name=worker_pool.name)
        else:
            workers = (
                worker.resolve_with_context(context=context).unwrap()
                for worker in worker_pool.worker_list
            )

            worker_ids = []
            for worker in workers:
                worker_ids.append(worker.id)

            for id_ in worker_ids:
                context.server.services.worker.delete(
                    context=context, uid=id_, force=True
                )

        worker_pool.max_count = 0
        worker_pool.worker_list = []
        self.stash.update(
            credentials=context.credentials,
            obj=worker_pool,
        ).unwrap(
            public_message=(
                f"Pool {worker_pool.name} was purged, "
                f"but failed to update the stash"
            )
        )

        return SyftSuccess(message=f"Successfully Purged worker pool with id {uid}")

    @as_result(StashException, SyftException)
    def _get_worker_pool(
        self,
        context: AuthedServiceContext,
        pool_id: UID | None = None,
        pool_name: str | None = None,
    ) -> WorkerPool:
        if pool_id:
            worker_pool = self.stash.get_by_uid(
                credentials=context.credentials,
                uid=pool_id,
            ).unwrap()
        else:
            worker_pool = self.stash.get_by_name(
                credentials=context.credentials,
                pool_name=pool_name,
            ).unwrap()

        return worker_pool


@as_result(SyftException)
def _create_workers_in_pool(
    context: AuthedServiceContext,
    pool_name: str,
    existing_worker_cnt: int,
    worker_cnt: int,
    worker_image: SyftWorkerImage,
    worker_stash: WorkerStash,
    registry_username: str | None = None,
    registry_password: str | None = None,
    pod_annotations: dict[str, str] | None = None,
    pod_labels: dict[str, str] | None = None,
) -> tuple[list[LinkedObject], list[ContainerSpawnStatus]]:
    queue_port = context.server.queue_config.client_config.queue_port

    # Check if workers needs to be run in memory or as containers
    start_workers_in_memory = context.server.in_memory_workers

    if start_workers_in_memory:
        # Run in-memory workers in threads
        container_statuses: list[ContainerSpawnStatus] = run_workers_in_threads(
            server=context.server,
            pool_name=pool_name,
            start_idx=existing_worker_cnt,
            number=worker_cnt + existing_worker_cnt,
        )
    else:
        registry_host = (
            worker_image.image_identifier.registry_host
            if worker_image.image_identifier is not None
            else None
        )
        container_statuses = run_containers(
            pool_name=pool_name,
            worker_image=worker_image,
            start_idx=existing_worker_cnt,
            number=worker_cnt + existing_worker_cnt,
            orchestration=get_orchestration_type(),
            queue_port=queue_port,
            dev_mode=context.server.dev_mode,
            registry_username=registry_username,
            registry_password=registry_password,
            reg_url=registry_host,
            pod_annotations=pod_annotations,
            pod_labels=pod_labels,
        ).unwrap()

    linked_worker_list = []

    for container_status in container_statuses:
        worker = container_status.worker
        if worker is None:
            continue

        server = context.server

        try:
            obj = worker_stash.set(
                credentials=context.credentials,
                obj=worker,
            ).unwrap()

            worker_obj = LinkedObject.from_obj(
                obj=obj,
                service_type=WorkerService,
                server_uid=server.id,
            )

            linked_worker_list.append(worker_obj)
        except SyftException as exc:
            container_status.error = exc.public_message
    return linked_worker_list, container_statuses


TYPE_TO_SERVICE[WorkerPool] = SyftWorkerPoolService
SERVICE_TO_TYPES[SyftWorkerPoolService] = WorkerPool
