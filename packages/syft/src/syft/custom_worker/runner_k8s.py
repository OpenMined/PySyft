# stdlib
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

# third party
from kr8s.objects import Pod
from kr8s.objects import Secret
from kr8s.objects import StatefulSet

# relative
from .k8s import KUBERNETES_NAMESPACE
from .k8s import KubeUtils
from .k8s import PodStatus
from .k8s import get_kr8s_client

JSONPATH_AVAILABLE_REPLICAS = "{.status.availableReplicas}"
CREATE_POOL_TIMEOUT_SEC = 60
SCALE_POOL_TIMEOUT_SEC = 60


class KubernetesRunner:
    def __init__(self):
        self.client = get_kr8s_client()

    def create_pool(
        self,
        pool_name: str,
        tag: str,
        replicas: int = 1,
        env_vars: Optional[List[Dict]] = None,
        mount_secrets: Optional[Dict] = None,
        reg_username: Optional[str] = None,
        reg_password: Optional[str] = None,
        reg_url: Optional[str] = None,
        **kwargs,
    ) -> StatefulSet:
        try:
            # create pull secret if registry credentials are passed
            pull_secret = None
            if reg_username and reg_password and reg_url:
                pull_secret = self._create_image_pull_secret(
                    pool_name,
                    reg_username,
                    reg_password,
                    reg_url,
                )

            # create a stateful set deployment
            deployment = self._create_stateful_set(
                pool_name=pool_name,
                tag=tag,
                replicas=replicas,
                env_vars=env_vars,
                mount_secrets=mount_secrets,
                pull_secret=pull_secret,
                **kwargs,
            )

            # wait for replicas to be available and ready
            deployment.wait(
                f"jsonpath='{JSONPATH_AVAILABLE_REPLICAS}'={replicas}",
                timeout=CREATE_POOL_TIMEOUT_SEC,
            )
        except Exception:
            raise
        finally:
            if pull_secret:
                pull_secret.delete(propagation_policy="Foreground")

        # return
        return deployment

    def scale_pool(self, pool_name: str, replicas: int) -> Optional[StatefulSet]:
        deployment = self.get_pool(pool_name)
        if not deployment:
            return None
        deployment.scale(replicas)
        deployment.wait(
            f"jsonpath='{JSONPATH_AVAILABLE_REPLICAS}'={replicas}",
            timeout=SCALE_POOL_TIMEOUT_SEC,
        )
        return deployment

    def exists(self, pool_name: str) -> bool:
        return bool(self.get_pool(pool_name))

    def get_pool(self, pool_name: str) -> Optional[StatefulSet]:
        selector = {"app.kubernetes.io/component": pool_name}
        for _set in self.client.get("statefulsets", label_selector=selector):
            return _set
        return None

    def delete_pool(self, pool_name: str) -> bool:
        selector = {"app.kubernetes.io/component": pool_name}
        for _set in self.client.get("statefulsets", label_selector=selector):
            _set.delete(propagation_policy="Foreground")

        for _secret in self.client.get("secrets", label_selector=selector):
            _secret.delete(propagation_policy="Foreground")

        return True

    def delete_pod(self, pod_name: str) -> bool:
        pods = self.client.get("pods", pod_name)
        for pod in pods:
            pod.delete(propagation_policy="Foreground")
            return True
        return False

    def get_pool_pods(self, pool_name: str) -> List[Pod]:
        selector = {"app.kubernetes.io/component": pool_name}
        pods = self.client.get("pods", label_selector=selector)
        if len(pods) > 0:
            pods.sort(key=lambda pod: pod.name)
        return pods

    def get_pod_logs(self, pod_name: str) -> str:
        pods = self.client.get("pods", pod_name)
        return KubeUtils.get_logs(pods)

    def get_pod_status(self, pod: Union[str, Pod]) -> Optional[PodStatus]:
        pod = KubeUtils.resolve_pod(self.client, pod)
        return KubeUtils.get_pod_status(pod)

    def get_pod_env_vars(self, pod: Union[str, Pod]) -> Optional[List[Dict]]:
        pod = KubeUtils.resolve_pod(self.client, pod)
        return KubeUtils.get_pod_env(pod)

    def _create_image_pull_secret(
        self,
        pool_name: str,
        reg_username: str,
        reg_password: str,
        reg_url: str,
        **kwargs,
    ):
        return KubeUtils.create_dockerconfig_secret(
            secret_name=f"pull-secret-{pool_name}",
            component=pool_name,
            registries=[
                (reg_url, reg_username, reg_password),
            ],
        )

    def _create_stateful_set(
        self,
        pool_name: str,
        tag: str,
        replicas=1,
        env_vars: Optional[List[Dict]] = None,
        mount_secrets: Optional[Dict] = None,
        pull_secret: Optional[Secret] = None,
        **kwargs,
    ) -> StatefulSet:
        """Create a stateful set for a pool"""

        volumes = []
        volume_mounts = []
        pull_secret_obj = None
        env_vars = env_vars or []

        if mount_secrets:
            for secret_name, mount_opts in mount_secrets.items():
                volumes.append(
                    {
                        "name": secret_name,
                        "secret": {
                            "secretName": secret_name,
                        },
                    }
                )
                volume_mounts.append(
                    {
                        "name": secret_name,
                        "mountPath": mount_opts.get("mountPath"),
                        "subPath": mount_opts.get("subPath"),
                        "readOnly": True,
                    }
                )

        if pull_secret:
            pull_secret_obj = [
                {
                    "name": pull_secret.name,
                }
            ]

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
                                    "imagePullPolicy": "IfNotPresent",
                                    "image": tag,
                                    "env": env_vars,
                                    "volumeMounts": volume_mounts,
                                }
                            ],
                            "volumes": volumes,
                            "imagePullSecrets": pull_secret_obj,
                        },
                    },
                },
            }
        )
        return KubeUtils.create_or_get(stateful_set)
