# stdlib
from typing import Any

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
CREATE_POOL_TIMEOUT_SEC = 380
SCALE_POOL_TIMEOUT_SEC = 60


class KubernetesRunner:
    def __init__(self) -> None:
        self.client = get_kr8s_client()

    def create_pool(
        self,
        pool_name: str,
        tag: str,
        replicas: int = 1,
        env_vars: list[dict] | None = None,
        mount_secrets: dict | None = None,
        registry_username: str | None = None,
        registry_password: str | None = None,
        reg_url: str | None = None,
        pod_annotations: dict[str, str] | None = None,
        pod_labels: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> StatefulSet:
        try:
            # create pull secret if registry credentials are passed
            pull_secret = None
            if registry_username and registry_password and reg_url:
                pull_secret = self._create_image_pull_secret(
                    pool_name,
                    registry_username,
                    registry_password,
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
                pod_annotations=pod_annotations,
                pod_labels=pod_labels,
                **kwargs,
            )

            # wait for replicas to be available and ready
            deployment.wait(
                f"jsonpath='{JSONPATH_AVAILABLE_REPLICAS}'={replicas}",
                timeout=CREATE_POOL_TIMEOUT_SEC,
            )
        finally:
            if pull_secret:
                pull_secret.delete(propagation_policy="Foreground")

        # return
        return deployment

    def scale_pool(self, pool_name: str, replicas: int) -> StatefulSet | None:
        deployment = self.get_pool(pool_name)
        timeout = max(SCALE_POOL_TIMEOUT_SEC * replicas, SCALE_POOL_TIMEOUT_SEC)
        if not deployment:
            return None
        deployment.scale(replicas)
        deployment.wait(
            f"jsonpath='{JSONPATH_AVAILABLE_REPLICAS}'={replicas}",
            timeout=timeout,
        )
        return deployment

    def exists(self, pool_name: str) -> bool:
        return bool(self.get_pool(pool_name))

    def get_pool(self, pool_name: str) -> StatefulSet | None:
        selector = {"app.kubernetes.io/component": pool_name}
        for _set in self.client.get("statefulsets", label_selector=selector):
            return _set
        return None

    def delete_pool(self, pool_name: str) -> bool:
        selector = {"app.kubernetes.io/component": pool_name}
        for _set in self.client.get("statefulsets", label_selector=selector):
            _set.delete(propagation_policy="Foreground")
            _set.wait(conditions="delete")

        for _secret in self.client.get("secrets", label_selector=selector):
            _secret.delete(propagation_policy="Foreground")
            _secret.wait(conditions="delete")

        return True

    def delete_pod(self, pod_name: str) -> bool:
        pods = self.client.get("pods", pod_name)
        for pod in pods:
            pod.delete(propagation_policy="Foreground")
            return True
        return False

    def get_pool_pods(self, pool_name: str) -> list[Pod]:
        selector = {"app.kubernetes.io/component": pool_name}
        pods = self.client.get("pods", label_selector=selector)
        if len(pods) > 0:
            pods.sort(key=lambda pod: pod.name)
        return pods

    def get_pod_logs(self, pod_name: str) -> str:
        pods = self.client.get("pods", pod_name)
        return KubeUtils.get_logs(pods)

    def get_pod_status(self, pod: str | Pod) -> PodStatus | None:
        pod = KubeUtils.resolve_pod(self.client, pod)
        return KubeUtils.get_pod_status(pod)

    def get_pod_env_vars(self, pod: str | Pod) -> list[dict] | None:
        pod = KubeUtils.resolve_pod(self.client, pod)
        return KubeUtils.get_pod_env(pod)

    def _create_image_pull_secret(
        self,
        pool_name: str,
        registry_username: str,
        registry_password: str,
        reg_url: str,
        **kwargs: Any,
    ) -> Secret:
        return KubeUtils.create_dockerconfig_secret(
            secret_name=f"pull-secret-{pool_name}",
            component=pool_name,
            registries=[
                (reg_url, registry_username, registry_password),
            ],
        )

    def _create_stateful_set(
        self,
        pool_name: str,
        tag: str,
        replicas: int = 1,
        env_vars: list[dict] | None = None,
        mount_secrets: dict | None = None,
        pull_secret: Secret | None = None,
        pod_annotations: dict[str, str] | None = None,
        pod_labels: dict[str, str] | None = None,
        **kwargs: Any,
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

        default_pod_labels = {
            "app.kubernetes.io/name": KUBERNETES_NAMESPACE,
            "app.kubernetes.io/component": pool_name,
        }

        if isinstance(pod_labels, dict):
            pod_labels = {**default_pod_labels, **pod_labels}
        else:
            pod_labels = default_pod_labels

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
                            "labels": pod_labels,
                            "annotations": pod_annotations,
                        },
                        "spec": {
                            # TODO: make this configurable
                            "serviceAccountName": "backend-service-account",
                            "containers": [
                                {
                                    "name": pool_name,
                                    "imagePullPolicy": "IfNotPresent",
                                    "image": tag,
                                    "env": env_vars,
                                    "volumeMounts": volume_mounts,
                                    "livenessProbe": {
                                        "httpGet": {
                                            "path": "/api/v2/metadata?probe=livenessProbe",
                                            "port": 80,
                                        },
                                        "periodSeconds": 15,
                                        "timeoutSeconds": 5,
                                        "failureThreshold": 3,
                                    },
                                    "startupProbe": {
                                        "httpGet": {
                                            "path": "/api/v2/metadata?probe=startupProbe",
                                            "port": 80,
                                        },
                                        "failureThreshold": 30,
                                        "periodSeconds": 10,
                                    },
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
