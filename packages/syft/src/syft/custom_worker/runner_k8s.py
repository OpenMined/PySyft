# stdlib
import copy
import os
from time import sleep
from typing import List
from typing import Optional
from typing import Union

# third party
import kr8s
from kr8s.objects import APIObject
from kr8s.objects import Pod
from kr8s.objects import StatefulSet

# relative
from .k8s import KUBERNETES_NAMESPACE
from .k8s import PodStatus


class KubernetesRunner:
    def __init__(self):
        self.client = kr8s.api(namespace=KUBERNETES_NAMESPACE)

    def create_pool(
        self,
        pool_name: str,
        tag: str,
        replicas: int = 1,
        env_vars: Optional[dict] = None,
        **kwargs,
    ) -> StatefulSet:
        deployment = self._create_stateful_set(
            pool_name,
            tag,
            replicas,
            env_vars,
            **kwargs,
        )
        self.wait(deployment, available_replicas=replicas)
        return deployment

    def scale_pool(self, pool_name: str, replicas: int) -> Optional[StatefulSet]:
        deployment = self.get_pool(pool_name)
        if not deployment:
            return None
        deployment.scale(replicas)
        self.wait(deployment, available_replicas=replicas)
        return deployment

    def get_pool(self, pool_name: str) -> Optional[StatefulSet]:
        selector = {"app.kubernetes.io/component": pool_name}
        for _set in self.client.get("statefulsets", label_selector=selector):
            return _set
        return None

    def delete_pool(self, pool_name: str) -> bool:
        selector = {"app.kubernetes.io/component": pool_name}
        for _set in self.client.get("statefulsets", label_selector=selector):
            _set.delete()
            return True
        return False

    def delete_pod(self, pod_name: str) -> bool:
        pods = self.client.get("pods", pod_name)
        for pod in pods:
            pod.delete()
            return True
        return False

    def get_pods(self, pool_name: str) -> List[Pod]:
        selector = {"app.kubernetes.io/component": pool_name}
        pods = self.client.get("pods", label_selector=selector)
        if len(pods) > 0:
            pods.sort(key=lambda pod: pod.name)
        return pods

    def get_pod_logs(self, pod_name: str) -> str:
        pods = self.client.get("pods", pod_name)
        logs = []
        for pod in pods:
            logs.append(f"----------Logs for pod={pod.metadata.name}----------")
            for log in pod.logs():
                logs.append(log)

        return "\n".join(logs)

    def get_pod_status(self, pod: Union[str, Pod]) -> Optional[PodStatus]:
        if isinstance(pod, str):
            pods = self.client.get("pods", pod)
            if len(pods) == 0:
                return None
            pod = pods[0]
        else:
            pod.refresh()

        return PodStatus.from_status_dict(pod.status)

    def wait(
        self,
        deployment: StatefulSet,
        available_replicas: int,
        timeout: int = 60,
    ) -> None:
        # TODO: Report wait('jsonpath=') bug to kr8s
        # Until then this is the substitute implementation

        if available_replicas <= 0:
            return

        while True:
            if timeout == 0:
                raise TimeoutError("Timeout waiting for replicas")

            deployment.refresh()
            if deployment.status.availableReplicas == available_replicas:
                break

            timeout -= 1
            sleep(1)

    def _current_pod_name(self) -> str:
        env_val = os.getenv("K8S_POD_NAME")
        if env_val:
            return env_val

        selector = {"app.kubernetes.io/component": "backend"}
        for pod in self.client.get("pods", label_selector=selector):
            return pod.name

    def _get_obj_from_list(self, objs: List[dict], name: str) -> dict:
        """Helper function extract kubernetes object from list by name"""
        for obj in objs:
            if obj.name == name:
                return obj

    def _create_stateful_set(
        self,
        pool_name: str,
        tag: str,
        replicas=1,
        env_vars: Optional[dict] = None,
        **kwargs,
    ) -> StatefulSet:
        """Create a stateful set for a pool"""

        env_vars = env_vars or {}

        _pod = Pod.get(self._current_pod_name())

        creds_volume = self._get_obj_from_list(
            objs=_pod.spec.volumes,
            name="credentials-data",
        )
        creds_volume_mount = self._get_obj_from_list(
            objs=_pod.spec.containers[0].volumeMounts,
            name="credentials-data",
        )

        env = _pod.spec.containers[0].env.to_list()
        env_clone = copy.deepcopy(env)

        # update existing
        for item in env_clone:
            k = item["name"]
            if k in env_vars:
                v = env_vars.pop(k)
                item["value"] = v

        # append remaining
        for k, v in env_vars.items():
            env_clone.append({"name": k, "value": v})

        stateful_set = StatefulSet(
            {
                "metadata": {
                    "name": pool_name,
                    "labels": {
                        "app.kubernetes.io/name": KUBERNETES_NAMESPACE,
                        "app.kubernetes.io/component": pool_name,
                        "app.kubernetes.io/managed-by": "kr8s",
                    },
                },
                "spec": {
                    "replicas": replicas,
                    "selector": {
                        "matchLabels": {
                            "app.kubernetes.io/component": pool_name,
                        }
                    },
                    "template": {
                        "metadata": {
                            "labels": {
                                "app.kubernetes.io/name": KUBERNETES_NAMESPACE,
                                "app.kubernetes.io/component": pool_name,
                            }
                        },
                        "spec": {
                            "containers": [
                                {
                                    "name": pool_name,
                                    "image": tag,
                                    "env": env_clone,
                                    "volumeMounts": [creds_volume_mount],
                                }
                            ],
                            "volumes": [creds_volume],
                        },
                    },
                },
            }
        )
        return self._create_or_get(stateful_set)

    def _create_or_get(self, obj: APIObject) -> APIObject:
        if not obj.exists():
            obj.create()
        else:
            obj.refresh()
        return obj
