# stdlib
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

# third party
import pydantic

# relative
from ...custom_worker.config import CustomWorkerConfig
from ...custom_worker.config import WorkerConfig
from ...custom_worker.k8s import IN_KUBERNETES
from ...custom_worker.runner_k8s import KubernetesRunner
from ...serde.serializable import serializable
from ...store.document_store import DocumentStore
from ...store.linked_obj import LinkedObject
from ...types.dicttuple import DictTuple
from ...types.uid import UID
from ..context import AuthedServiceContext
from ..request.request import Change
from ..request.request import CreateCustomImageChange
from ..request.request import CreateCustomWorkerPoolChange
from ..request.request import Request
from ..request.request import SubmitRequest
from ..request.request_service import RequestService
from ..response import SyftError
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


@serializable()
class SyftWorkerPoolService(AbstractService):
    store: DocumentStore
    stash: SyftWorkerPoolStash

    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.stash = SyftWorkerPoolStash(store=store)
        self.image_stash = SyftWorkerImageStash(store=store)

    @service_method(
        path="worker_pool.launch",
        name="launch",
        roles=DATA_OWNER_ROLE_LEVEL,
    )
    def launch(
        self,
        context: AuthedServiceContext,
        name: str,
        image_uid: Optional[UID],
        num_workers: int,
        reg_username: Optional[str] = None,
        reg_password: Optional[str] = None,
    ) -> Union[List[ContainerSpawnStatus], SyftError]:
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

        result = self.stash.get_by_name(context.credentials, pool_name=name)

        if result.is_err():
            return SyftError(message=f"{result.err()}")

        if result.ok() is not None:
            return SyftError(message=f"Worker Pool with name: {name} already exists !!")

        # If image uid is not passed, then use the default worker image
        # to create the worker pool
        if image_uid is None:
            result = self.stash.get_by_name(
                context.credentials, pool_name=DEFAULT_WORKER_POOL_NAME
            )
            default_worker_pool = result.ok()
            image_uid = default_worker_pool.image_id

        # Get the image object for the given image id
        result = self.image_stash.get_by_uid(
            credentials=context.credentials, uid=image_uid
        )
        if result.is_err():
            return SyftError(
                message=f"Failed to retrieve Worker Image with id: {image_uid}. Error: {result.err()}"
            )

        worker_image: SyftWorkerImage = result.ok()

        worker_service: WorkerService = context.node.get_service("WorkerService")
        worker_stash = worker_service.stash

        # Create worker pool from given image, with the given worker pool
        # and with the desired number of workers
        result = _create_workers_in_pool(
            context=context,
            pool_name=name,
            existing_worker_cnt=0,
            worker_cnt=num_workers,
            worker_image=worker_image,
            worker_stash=worker_stash,
            reg_username=reg_username,
            reg_password=reg_password,
        )

        if isinstance(result, SyftError):
            return result

        worker_list, container_statuses = result

        # Update the Database with the pool information
        worker_pool = WorkerPool(
            name=name,
            max_count=num_workers,
            image_id=worker_image.id,
            worker_list=worker_list,
            syft_node_location=context.node.id,
            syft_client_verify_key=context.credentials,
        )
        result = self.stash.set(credentials=context.credentials, obj=worker_pool)

        if result.is_err():
            return SyftError(message=f"Failed to save Worker Pool: {result.err()}")

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
        reason: Optional[str] = "",
    ) -> Union[SyftError, SyftSuccess]:
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
        search_result = self.image_stash.get_by_uid(
            credentials=context.credentials, uid=image_uid
        )

        if search_result.is_err():
            return SyftError(message=str(search_result.err()))

        worker_image: Optional[SyftWorkerImage] = search_result.ok()

        # Raise error if worker image doesn't exists
        if worker_image is None:
            return SyftError(
                message=f"No image exists for given image uid : {image_uid}"
            )

        # Check if pool already exists for the given pool name
        result = self.stash.get_by_name(context.credentials, pool_name=pool_name)

        if result.is_err():
            return SyftError(message=f"{result.err()}")

        worker_pool = result.ok()

        if worker_pool is not None:
            return SyftError(
                message=f"Worker pool already exists for given pool name: {pool_name}"
            )

        # If no worker pool exists for given pool name
        # and image exists for given image uid, then create a change
        # request object to create the pool with the desired number of workers
        create_worker_pool_change = CreateCustomWorkerPoolChange(
            pool_name=pool_name,
            num_workers=num_workers,
            image_uid=image_uid,
        )

        changes: List[Change] = [create_worker_pool_change]

        # Create a the request object with the changes and submit it
        # for approval.
        request = SubmitRequest(changes=changes)
        method = context.node.get_service_method(RequestService.submit)
        result = method(context=context, request=request, reason=reason)

        return result

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
        tag: str,
        config: WorkerConfig,
        registry_uid: Optional[UID] = None,
        reason: Optional[str] = "",
    ) -> Union[SyftError, SyftSuccess]:
        """
        Create a request to launch the worker pool based on a built image.

        Args:
            context (AuthedServiceContext): The authenticated service context.
            pool_name (str): The name of the worker pool.
            num_workers (int): The number of workers in the pool.
            config: (WorkerConfig): Config of the image to be built.
            tag (str): human-readable manifest identifier that is typically a specific version or variant of an image
            reason (Optional[str], optional): The reason for creating the worker image and pool. Defaults to "".
        """

        if isinstance(config, CustomWorkerConfig):
            return SyftError(message="We only support DockerWorkerConfig.")

        if IN_KUBERNETES and registry_uid is None:
            return SyftError(message="Registry UID is required in Kubernetes mode.")

        # Check if an image already exists for given docker config
        search_result = self.image_stash.get_by_docker_config(
            credentials=context.credentials, config=config
        )

        if search_result.is_err():
            return SyftError(message=str(search_result.err()))

        worker_image: Optional[SyftWorkerImage] = search_result.ok()

        if worker_image is not None:
            return SyftError(
                message="Image already exists for given config. \
                    Please use `worker_pool.create_pool_request` to request pool creation."
            )

        # Validate Image Tag
        try:
            SyftWorkerImageIdentifier.from_str(tag=tag)
        except pydantic.ValidationError as e:
            return SyftError(message=f"Failed to create tag: {e}")

        # create a list of Change objects and submit a
        # request for these changes for approval
        changes: List[Change] = []

        # Add create custom image change
        # If this change is approved, then build an image using the config
        create_custom_image_change = CreateCustomImageChange(
            config=config,
            tag=tag,
            registry_uid=registry_uid,
        )

        # Check if a pool already exists for given pool name
        result = self.stash.get_by_name(context.credentials, pool_name=pool_name)

        if result.is_err():
            return SyftError(message=f"{result.err()}")

        # Raise an error if worker pool already exists for the given worker pool name
        if result.ok() is not None:
            return SyftError(
                message=f"Worker Pool with name: {pool_name} already "
                f"exists. Please choose another name!"
            )

        # Add create worker pool change
        # If change is approved then worker pool is created and
        # the desired number of workers are added to the pool
        create_worker_pool_change = CreateCustomWorkerPoolChange(
            pool_name=pool_name,
            num_workers=num_workers,
            config=config,
        )
        changes += [create_custom_image_change, create_worker_pool_change]

        # Create a request object and submit a request for approval
        request = SubmitRequest(changes=changes)
        method = context.node.get_service_method(RequestService.submit)
        result = method(context=context, request=request, reason=reason)

        return result

    @service_method(
        path="worker_pool.get_all",
        name="get_all",
        roles=DATA_SCIENTIST_ROLE_LEVEL,
    )
    def get_all(
        self, context: AuthedServiceContext
    ) -> Union[DictTuple[str, WorkerPool], SyftError]:
        # TODO: During get_all, we should dynamically make a call to docker to get the status of the containers
        # and update the status of the workers in the pool.
        result = self.stash.get_all(credentials=context.credentials)
        if result.is_err():
            return SyftError(message=f"{result.err()}")
        worker_pools: List[WorkerPool] = result.ok()

        res: List[Tuple] = []
        for pool in worker_pools:
            res.append((pool.name, pool))
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
        pool_id: Optional[UID] = None,
        pool_name: Optional[str] = None,
        reg_username: Optional[str] = None,
        reg_password: Optional[str] = None,
    ) -> Union[List[ContainerSpawnStatus], SyftError]:
        """Add workers to existing worker pool.

        Worker pool is fetched either using the unique pool id or pool name.

        Args:
            context (AuthedServiceContext): _description_
            number (int): number of workers to add
            pool_id (Optional[UID], optional): Unique UID of the pool. Defaults to None.
            pool_name (Optional[str], optional): Unique name of the pool. Defaults to None.

        Returns:
            Union[List[ContainerSpawnStatus], SyftError]: List of spawned workers with their status and error if any.
        """

        if number <= 0:
            return SyftError(message=f"Invalid number of workers: {number}")

        # Extract pool using either using pool id or pool name
        if pool_id:
            result = self.stash.get_by_uid(credentials=context.credentials, uid=pool_id)
        elif pool_name:
            result = self.stash.get_by_name(
                credentials=context.credentials,
                pool_name=pool_name,
            )

        if result.is_err():
            return SyftError(message=f"{result.err()}")

        worker_pool = result.ok()

        existing_worker_cnt = len(worker_pool.worker_list)

        result = self.image_stash.get_by_uid(
            credentials=context.credentials,
            uid=worker_pool.image_id,
        )

        if result.is_err():
            return SyftError(
                message=f"Failed to retrieve image for worker pool: {worker_pool.name}"
            )

        worker_image: SyftWorkerImage = result.ok()

        worker_service: WorkerService = context.node.get_service("WorkerService")
        worker_stash = worker_service.stash

        # Add workers to given pool from the given image
        result = _create_workers_in_pool(
            context=context,
            pool_name=worker_pool.name,
            existing_worker_cnt=existing_worker_cnt,
            worker_cnt=number,
            worker_image=worker_image,
            worker_stash=worker_stash,
            reg_username=reg_username,
            reg_password=reg_password,
        )

        if isinstance(result, SyftError):
            return result

        worker_list, container_statuses = result

        worker_pool.worker_list += worker_list
        worker_pool.max_count = existing_worker_cnt + number

        update_result = self.stash.update(
            credentials=context.credentials, obj=worker_pool
        )
        if update_result.is_err():
            return SyftError(
                message=f"Failed update worker pool: {worker_pool.name} with err: {result.err()}"
            )

        return container_statuses

    @service_method(
        path="worker_pool.scale",
        name="scale",
        roles=DATA_OWNER_ROLE_LEVEL,
    )
    def scale(
        self,
        context: AuthedServiceContext,
        number: int,
        pool_id: Optional[UID] = None,
        pool_name: Optional[str] = None,
    ) -> Union[SyftError, SyftSuccess]:
        """
        Scale the worker pool to the given number of workers in Kubernetes.
        Allows both scaling up and down the worker pool.
        """

        if not IN_KUBERNETES:
            return SyftError(message="Scaling is only supported in Kubernetes mode")
        elif number < 0:
            # zero is a valid scale down
            return SyftError(message=f"Invalid number of workers: {number}")

        result = self._get_worker_pool(context, pool_id, pool_name)
        if isinstance(result, SyftError):
            return result

        worker_pool = result
        current_worker_count = len(worker_pool.worker_list)

        if current_worker_count == number:
            return SyftSuccess(message=f"Worker pool already has {number} workers")
        elif number > current_worker_count:
            workers_to_add = number - current_worker_count
            result = self.add_workers(
                context=context,
                number=workers_to_add,
                pool_id=pool_id,
                pool_name=pool_name,
                # kube scaling doesn't require password as it replicates an existing deployment
                reg_username=None,
                reg_password=None,
            )
            if isinstance(result, SyftError):
                return result
        else:
            # scale down at kubernetes control plane
            runner = KubernetesRunner()
            result = scale_kubernetes_pool(
                runner,
                pool_name=worker_pool.name,
                replicas=number,
            )
            if isinstance(result, SyftError):
                return result

            # scale down removes the last "n" workers
            # workers to delete = len(workers) - number
            workers_to_delete = worker_pool.worker_list[
                -(current_worker_count - number) :
            ]

            worker_stash = context.node.get_service("WorkerService").stash
            # delete linkedobj workers
            for worker in workers_to_delete:
                delete_result = worker_stash.delete_by_uid(
                    credentials=context.credentials,
                    uid=worker.object_uid,
                )
                if delete_result.is_err():
                    print(f"Failed to delete worker: {worker.object_uid}")

            # update worker_pool
            worker_pool.max_count = number
            worker_pool.worker_list = worker_pool.worker_list[:number]
            update_result = self.stash.update(
                credentials=context.credentials,
                obj=worker_pool,
            )

            if update_result.is_err():
                return SyftError(
                    message=(
                        f"Pool {worker_pool.name} was scaled down, "
                        f"but failed update the stash with err: {update_result.err()}"
                    )
                )

        return SyftSuccess(message=f"Worker pool scaled to {number} workers")

    @service_method(
        path="worker_pool.filter_by_image_id",
        name="filter_by_image_id",
        roles=DATA_SCIENTIST_ROLE_LEVEL,
    )
    def filter_by_image_id(
        self, context: AuthedServiceContext, image_uid: UID
    ) -> Union[List[WorkerPool], SyftError]:
        result = self.stash.get_by_image_uid(context.credentials, image_uid)

        if result.is_err():
            return SyftError(message=f"Failed to get worker pool for uid: {image_uid}")

        return result.ok()

    @service_method(
        path="worker_pool.get_by_name",
        name="get_by_name",
        roles=DATA_SCIENTIST_ROLE_LEVEL,
    )
    def get_by_name(
        self, context: AuthedServiceContext, pool_name: str
    ) -> Union[List[WorkerPool], SyftError]:
        result = self.stash.get_by_name(context.credentials, pool_name)

        if result.is_err():
            return SyftError(
                message=f"Failed to get worker pool with name: {pool_name}"
            )

        return result.ok()

    @service_method(
        path="worker_pool.sync_pool_from_request",
        name="sync_pool_from_request",
        roles=DATA_SCIENTIST_ROLE_LEVEL,
    )
    def sync_pool_from_request(
        self,
        context: AuthedServiceContext,
        request: Request,
    ) -> Union[SyftSuccess, SyftError]:
        """Re-submit request from a different node"""

        num_of_changes = len(request.changes)
        pool_name, num_workers, config, image_uid, tag = None, None, None, None, None

        if num_of_changes > 2:
            return SyftError(
                message=f"Invalid pool request object. Only pool request changes allowed. {request.changes}"
            )

        for change in request.changes:
            if isinstance(change, CreateCustomWorkerPoolChange):
                pool_name = change.pool_name
                num_workers = change.num_workers
                image_uid = change.image_uid
            elif isinstance(change, CreateCustomImageChange):
                config = change.config
                tag = change.tag

        if config is None and image_uid is not None:
            return self.create_pool_request(
                context=context,
                pool_name=pool_name,
                num_workers=num_workers,
                image_uid=image_uid,
            )
        elif config is not None:
            return self.create_image_and_pool_request(
                context=context,
                pool_name=pool_name,
                num_workers=num_workers,
                config=config,
                tag=tag,
            )
        else:
            return SyftError(
                message=f"Invalid request object. Invalid image uid or config in the request changes. {request.changes}"
            )

    def _get_worker_pool(
        self,
        context: AuthedServiceContext,
        pool_id: Optional[UID] = None,
        pool_name: Optional[str] = None,
    ) -> Union[WorkerPool, SyftError]:
        if pool_id:
            result = self.stash.get_by_uid(
                credentials=context.credentials,
                uid=pool_id,
            )
        else:
            result = self.stash.get_by_name(
                credentials=context.credentials,
                pool_name=pool_name,
            )

        if result.is_err():
            return SyftError(message=f"{result.err()}")

        worker_pool = result.ok()

        return (
            SyftError(
                message=f"worker pool : {pool_id if pool_id else pool_name} does not exist"
            )
            if worker_pool is None
            else worker_pool
        )


def _create_workers_in_pool(
    context: AuthedServiceContext,
    pool_name: str,
    existing_worker_cnt: int,
    worker_cnt: int,
    worker_image: SyftWorkerImage,
    worker_stash: WorkerStash,
    reg_username: Optional[str] = None,
    reg_password: Optional[str] = None,
) -> Union[Tuple[List[LinkedObject], List[ContainerSpawnStatus]], SyftError]:
    queue_port = context.node.queue_config.client_config.queue_port

    # Check if workers needs to be run in memory or as containers
    start_workers_in_memory = context.node.in_memory_workers

    if start_workers_in_memory:
        # Run in-memory workers in threads
        container_statuses: List[ContainerSpawnStatus] = run_workers_in_threads(
            node=context.node,
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
        result = run_containers(
            pool_name=pool_name,
            worker_image=worker_image,
            start_idx=existing_worker_cnt,
            number=worker_cnt + existing_worker_cnt,
            orchestration=get_orchestration_type(),
            queue_port=queue_port,
            dev_mode=context.node.dev_mode,
            reg_username=reg_username,
            reg_password=reg_password,
            reg_url=registry_host,
        )
        if isinstance(result, SyftError):
            return result
        container_statuses = result

    linked_worker_list = []

    for container_status in container_statuses:
        worker = container_status.worker
        if worker is None:
            continue
        result = worker_stash.set(
            credentials=context.credentials,
            obj=worker,
        )

        if result.is_ok():
            worker_obj = LinkedObject.from_obj(
                obj=result.ok(),
                service_type=WorkerService,
                node_uid=context.node.id,
            )
            linked_worker_list.append(worker_obj)
        else:
            container_status.error = result.err()

    return linked_worker_list, container_statuses


TYPE_TO_SERVICE[WorkerPool] = SyftWorkerPoolService
SERVICE_TO_TYPES[SyftWorkerPoolService] = WorkerPool
