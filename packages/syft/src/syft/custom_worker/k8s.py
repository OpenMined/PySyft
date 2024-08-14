# stdlib
import base64
from collections.abc import Iterable
from enum import Enum
from functools import cache
import json
import os

# third party
import kr8s
from kr8s.objects import APIObject
from kr8s.objects import ConfigMap
from kr8s.objects import Pod
from kr8s.objects import Secret
from kr8s.objects import Service
from pydantic import BaseModel
from typing_extensions import Self

# Kubernetes namespace
KUBERNETES_NAMESPACE = os.getenv("K8S_NAMESPACE", "syft")

# Kubernetes runtime flag
IN_KUBERNETES = os.getenv("CONTAINER_HOST") == "k8s"

# skip pushing to internal registry
USE_INTERNAL_REGISTRY = os.getenv("USE_INTERNAL_REGISTRY", "true").lower() == "true"

# Kaniko version
KANIKO_VERSION = os.getenv("KANIKO_VERSION", "latest")

# Internal registry URL
DEFAULT_INTERNAL_REGISTRY = f"registry.{KUBERNETES_NAMESPACE}.svc.cluster.local"
INTERNAL_REGISTRY_HOST = os.getenv("INTERNAL_REGISTRY_HOST", DEFAULT_INTERNAL_REGISTRY)


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
    def from_conditions(cls, conditions: list) -> Self:
        pod_cond = KubeUtils.list_dict_unpack(conditions, key="type", value="status")
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
    reason: str | None = None  # when waiting=True
    message: str | None = None  # when waiting=True
    startedAt: str | None = None  # when running=True

    @classmethod
    def from_status(cls, cstatus: dict) -> Self:
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
    def from_status_dict(cls, status: dict) -> Self:
        return cls(
            phase=PodPhase(status.get("phase", "Unknown")),
            condition=PodCondition.from_conditions(status.get("conditions", [])),
            container=ContainerStatus.from_status(
                status.get("containerStatuses", {})[0]
            ),
        )


@cache
def get_kr8s_client() -> kr8s.Api:
    if not IN_KUBERNETES:
        raise RuntimeError("Not inside a kubernetes environment")
    return kr8s.api(namespace=KUBERNETES_NAMESPACE)


class KubeUtils:
    """
    This class contains utility functions for interacting with kubernetes objects.

    DO NOT call `get_kr8s_client()` inside this class, instead pass it as an argument to the functions.
    This is to avoid calling these functions on resources across namespaces!
    """

    @staticmethod
    def resolve_pod(client: kr8s.Api, pod: str | Pod) -> Pod | None:
        """Return the first pod that matches the given name"""
        if isinstance(pod, Pod):
            return pod

        for _pod in client.get("pods", pod):
            return _pod

        return None

    @staticmethod
    def get_logs(pods: list[Pod]) -> str:
        """Combine and return logs for all the pods as a single string."""
        return "\n".join(
            f"----------Logs for pod={pod.metadata.name}----------\n{''.join(pod.logs())}"
            for pod in pods
        )

    @staticmethod
    def get_pod_status(pod: Pod) -> PodStatus | None:
        """Map the status of the given pod to PodStatuss."""
        if not pod:
            return None
        return PodStatus.from_status_dict(pod.status)

    @staticmethod
    def get_pod_env(pod: Pod) -> list[dict] | None:
        """Return the environment variables of the first container in the pod."""
        if not pod:
            return None

        for container in pod.spec.containers:
            return container.env.to_list()

        return None

    @staticmethod
    def get_container_exit_code(pods: list[Pod]) -> list[int]:
        """Return the exit codes of all the containers in the given pods."""
        return [
            container_status.state.terminated.exitCode
            for pod in pods
            for container_status in pod.status.containerStatuses
        ]

    @staticmethod
    def get_container_exit_message(pods: list[Pod]) -> str | None:
        """Return the exit message of the first container that exited with non-zero code."""
        for pod in pods:
            for container_status in pod.status.containerStatuses:
                if container_status.state.terminated.exitCode != 0:
                    continue
                return container_status.state.terminated.message
        return None

    @staticmethod
    def b64encode_secret(data: str) -> str:
        """Convert the data to base64 encoded string for Secret."""
        return base64.b64encode(data.encode()).decode()

    @staticmethod
    def get_configmap(client: kr8s.Api, name: str) -> ConfigMap | None:
        config_map = client.get("configmaps", name)
        return config_map[0] if config_map else None

    @staticmethod
    def get_service(client: kr8s.Api, name: str) -> Service | None:
        service = client.get("services", name)
        return service[0] if service else None

    @staticmethod
    def update_configmap(
        config_map: ConfigMap,
        patch: dict,
    ) -> None:
        existing_data = config_map.raw
        existing_data.update(patch)
        config_map.patch(patch=existing_data)

    @staticmethod
    def create_dockerconfig_secret(
        secret_name: str,
        component: str,
        registries: Iterable[tuple[str, str, str]],
    ) -> Secret:
        auths = {}

        for url, uname, passwd in registries:
            auths[url] = {
                "username": uname,
                "password": passwd,
                "auth": KubeUtils.b64encode_secret(f"{uname}:{passwd}"),
            }

        config_str = json.dumps({"auths": auths})

        return KubeUtils.create_secret(
            secret_name=secret_name,
            type="kubernetes.io/dockerconfigjson",
            component=component,
            data={
                ".dockerconfigjson": KubeUtils.b64encode_secret(config_str),
            },
        )

    @staticmethod
    def create_secret(
        secret_name: str,
        type: str,
        component: str,
        data: str,
        encoded: bool = True,
    ) -> Secret:
        if not encoded:
            for k, v in data.items():
                data[k] = KubeUtils.b64encode_secret(v)  # type: ignore

        secret = Secret(
            {
                "metadata": {
                    "name": secret_name,
                    "labels": {
                        "app.kubernetes.io/name": KUBERNETES_NAMESPACE,
                        "app.kubernetes.io/component": component,
                        "app.kubernetes.io/managed-by": "kr8s",
                    },
                },
                "type": type,
                "data": data,
            }
        )
        return KubeUtils.create_or_get(secret)

    @staticmethod
    def create_or_get(obj: APIObject) -> APIObject:
        if obj.exists():
            obj.refresh()
        else:
            obj.create()
        return obj

    @staticmethod
    def patch_env_vars(env_list: list[dict], env_dict: dict) -> list[dict]:
        """Patch kubernetes pod environment variables in the list with the provided dictionary."""

        # update existing
        for item in env_list:
            k = item["name"]
            if k in env_dict:
                v = env_dict.pop(k)
                item["value"] = v

        # append remaining
        for k, v in env_dict.items():
            env_list.append({"name": k, "value": v})

        return env_list

    @staticmethod
    def list_dict_unpack(
        input_list: list[dict],
        key: str = "key",
        value: str = "value",
    ) -> dict:
        # Snapshot from kr8s._data_utils
        return {i[key]: i[value] for i in input_list}
