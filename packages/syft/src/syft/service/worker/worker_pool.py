# stdlib
from enum import Enum
from typing import List
from typing import Optional

# relative
from ...client.api import APIRegistry
from ...serde.serializable import serializable
from ...store.linked_obj import LinkedObject
from ...types.base import SyftBaseModel
from ...types.datetime import DateTime
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SyftObject
from ...types.syft_object import short_uid
from ...types.uid import UID


@serializable()
class WorkerStatus(Enum):
    PENDING = "Pending"
    RUNNING = "Running"
    STOPPED = "Stopped"
    RESTARTED = "Restarted"


@serializable()
class ConsumerState(Enum):
    IDLE = "Idle"
    CONSUMING = "Consuming"


@serializable()
class WorkerHealth(Enum):
    HEALTHY = "✅"
    UNHEALTHY = "❌"


@serializable()
class SyftWorker(SyftObject):
    __canonical_name__ = "SyftWorker"
    __version__ = SYFT_OBJECT_VERSION_1

    __attr_unique__ = ["name"]
    __attr_searchable__ = ["name", "container_id", "image_hash", "worker_pool_name"]

    id: UID
    name: str
    container_id: Optional[str]
    created_at: DateTime = DateTime.now()
    image_hash: Optional[str]
    healthcheck: Optional[WorkerHealth]
    status: WorkerStatus
    worker_pool_name: str
    consumer_state: ConsumerState = ConsumerState.IDLE
    job_id: Optional[UID]

    def get_job_repr(self):
        if self.job_id is not None:
            api = APIRegistry.api_for(
                node_uid=self.syft_node_location,
                user_verify_key=self.syft_client_verify_key,
            )
            job = api.services.job.get(self.job_id)
            if job.action.user_code_id is not None:
                func_name = api.services.code.get_by_id(
                    job.action.user_code_id
                ).service_func_name
                return f"{func_name} ({short_uid(self.job_id)})"
            else:
                return f"action ({short_uid(self.job_id)})"
        else:
            return ""

    def _coll_repr_(self):
        return {
            "name": self.name,
            "container id": self.container_id,
            "status": str(self.status.value.lower()),
            "job": self.get_job_repr(),
            "created at": str(self.created_at),
            "consumer state": str(self.consumer_state.value.lower()),
        }


@serializable()
class WorkerPool(SyftObject):
    __canonical_name__ = "WorkerPool"
    __version__ = SYFT_OBJECT_VERSION_1

    __attr_unique__ = ["name"]
    __attr_searchable__ = ["name", "syft_worker_image_id"]

    name: str
    syft_worker_image_id: UID
    max_count: int
    worker_list: List[LinkedObject]

    @property
    def workers(self):
        return [worker.resolve for worker in self.worker_list]


@serializable()
class WorkerOrchestrationType:
    DOCKER = "docker"
    K8s = "k8s"
    PYTHON = "python"


@serializable()
class ContainerSpawnStatus(SyftBaseModel):
    __repr_attrs__ = ["worker_name", "worker", "error"]

    worker_name: str
    worker: Optional[SyftWorker]
    error: Optional[str]
