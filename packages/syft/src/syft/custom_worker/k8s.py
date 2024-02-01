# stdlib
from enum import Enum
import os
from typing import Optional

# third party
from kr8s._data_utils import list_dict_unpack
from pydantic import BaseModel

# Time after which Job will be deleted
JOB_COMPLETION_TTL = 60

# Persistent volume claim for storing build output
BUILD_OUTPUT_PVC = "worker-builds"

# Kubernetes namespace
KUBERNETES_NAMESPACE = os.getenv("K8S_NAMESPACE", "syft")

# Kubernetes runtime flag
IN_KUBERNETES = os.getenv("CONTAINER_HOST") == "k8s"


class PodPhase(Enum):
    Pending = "Pending"
    Running = "Running"
    Succeeded = "Succeeded"
    Failed = "Failed"
    Unknown = "Unknown"


class PodCondition(BaseModel):
    pod_scheduled: bool
    containers_ready: bool
    initialized: bool
    ready: bool

    @classmethod
    def from_conditions(cls, conditions: list):
        pod_cond = list_dict_unpack(conditions, key="type", value="status")
        pod_cond_flags = {k: v == "True" for k, v in pod_cond.items()}
        return cls(
            pod_scheduled=pod_cond_flags.get("PodScheduled", False),
            containers_ready=pod_cond_flags.get("ContainersReady", False),
            initialized=pod_cond_flags.get("Initialized", False),
            ready=pod_cond_flags.get("Ready", False),
        )


class ContainerStatus(BaseModel):
    ready: bool
    running: bool
    waiting: bool
    reason: Optional[str]  # when waiting=True
    message: Optional[str]  # when waiting=True
    startedAt: Optional[str]  # when running=True

    @classmethod
    def from_status(cls, cstatus: dict):
        cstate = cstatus.get("state", {})

        return cls(
            ready=cstatus.get("ready", False),
            running="running" in cstate,
            waiting="waiting" in cstate,
            reason=cstate.get("waiting", {}).get("reason", None),
            message=cstate.get("waiting", {}).get("message", None),
            startedAt=cstate.get("running", {}).get("startedAt", None),
        )


class PodStatus(BaseModel):
    phase: PodPhase
    condition: PodCondition
    container: ContainerStatus

    @classmethod
    def from_status_dict(cls: "PodStatus", status: dict):
        return cls(
            phase=PodPhase(status.get("phase", "Unknown")),
            condition=PodCondition.from_conditions(status.get("conditions", [])),
            container=ContainerStatus.from_status(
                status.get("containerStatuses", {})[0]
            ),
        )
