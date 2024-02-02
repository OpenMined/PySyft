# stdlib
from hashlib import sha256
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional

# third party
import kr8s
from kr8s.objects import APIObject
from kr8s.objects import ConfigMap
from kr8s.objects import Job
from kr8s.objects import Secret

# relative
from .builder_types import BuilderBase
from .builder_types import ImageBuildResult
from .builder_types import ImagePushResult
from .k8s import INTERNAL_REGISTRY_HOST
from .k8s import JOB_COMPLETION_TTL
from .k8s import KUBERNETES_NAMESPACE
from .utils import create_dockerconfig_json
from .utils import parse_tag

__all__ = ["KubernetesBuilder"]


class BuildFailed(Exception):
    pass


class KubernetesBuilder(BuilderBase):
    def __init__(self):
        self.client = kr8s.api(namespace=KUBERNETES_NAMESPACE)

    def build_image(
        self,
        tag: str,
        dockerfile: str = None,
        dockerfile_path: Path = None,
        buildargs: Optional[dict] = None,
        **kwargs,
    ) -> ImageBuildResult:
        image_digest = None
        logs = None
        config = None
        job_id = self._new_job_id(tag)

        if dockerfile:
            pass
        elif dockerfile_path:
            dockerfile = dockerfile_path.read_text()

        try:
            # Create a ConfigMap with the Dockerfile
            config = self._create_build_config(job_id, dockerfile)
            config.refresh()

            # Create and start the job
            job = self._create_kaniko_build_job(
                job_id=job_id,
                tag=tag,
                job_config=config,
                build_args=buildargs,
            )

            # wait for job to complete/fail
            job.wait(["condition=Complete", "condition=Failed"])

            # get logs
            logs = self._get_logs(job)

            image_digest = self._get_image_digest(job)
            if not image_digest:
                exit_code = self._get_container_exit_code(job)
                raise BuildFailed(
                    "Failed to build the image. "
                    f"Kaniko exit code={exit_code}. "
                    f"Logs={logs}"
                )

        except Exception:
            raise
        finally:
            # don't delete the job, kubernetes will gracefully do that for us
            config and config.delete(propagation_policy="Background")

        return ImageBuildResult(
            image_hash=image_digest,
            logs=logs,
        )

    def push_image(
        self,
        tag: str,
        username: str,
        password: str,
        registry_url: str,
        **kwargs,
    ) -> ImagePushResult:
        exit_code = 1
        logs = None
        job_id = self._new_job_id(tag)
        push_secret = None

        try:
            push_secret = self._create_push_secret(
                id=job_id,
                url=registry_url,
                username=username,
                password=password,
            )
            push_secret.refresh()

            job = self._create_push_job(
                job_id=job_id,
                tag=tag,
                push_secret=push_secret,
            )

            job.wait(["condition=Complete", "condition=Failed"])
            exit_code = self._get_container_exit_code(job)[0]
            logs = self._get_logs(job)
        except Exception:
            raise
        finally:
            push_secret and push_secret.delete(propagation_policy="Background")

        return ImagePushResult(logs=logs, exit_code=exit_code)

    def _new_job_id(self, tag: str) -> str:
        return self._get_tag_hash(tag)[:16]

    def _get_tag_hash(self, tag: str) -> str:
        return sha256(tag.encode()).hexdigest()

    def _get_image_digest(self, job: Job) -> Optional[str]:
        selector = {"batch.kubernetes.io/job-name": job.metadata.name}
        pods = self.client.get("pods", label_selector=selector)
        for pod in pods:
            for container_status in pod.status.containerStatuses:
                if container_status.state.terminated.exitCode != 0:
                    continue
                return container_status.state.terminated.message
        return None

    def _get_container_exit_code(self, job: Job) -> List[int]:
        selector = {"batch.kubernetes.io/job-name": job.metadata.name}
        pods = self.client.get("pods", label_selector=selector)
        exit_codes = []
        for pod in pods:
            for container_status in pod.status.containerStatuses:
                exit_codes.append(container_status.state.terminated.exitCode)
        return exit_codes

    def _get_logs(self, job: Job) -> str:
        selector = {"batch.kubernetes.io/job-name": job.metadata.name}
        pods = self.client.get("pods", label_selector=selector)
        logs = []
        for pod in pods:
            logs.append(f"----------Logs for pod={pod.metadata.name}----------")
            for log in pod.logs():
                logs.append(log)

        return "\n".join(logs)

    def _create_build_config(self, job_id: str, dockerfile: str) -> ConfigMap:
        config_map = ConfigMap(
            {
                "metadata": {
                    "name": f"build-{job_id}",
                },
                "data": {
                    "Dockerfile": dockerfile,
                },
            }
        )
        return self._create_or_get(config_map)

    def _change_registry(self, old_tag: str, registry: str):
        _, repo, tag = parse_tag(old_tag)
        new_tag = f"{registry}/{repo}:{tag}"
        print(f"changed from {old_tag} to {new_tag}")
        return new_tag

    def _create_kaniko_build_job(
        self,
        job_id: str,
        tag: str,
        job_config: ConfigMap,
        build_args: Optional[Dict] = None,
    ) -> Job:
        # for push
        build_args = build_args or {}
        build_args_list = []

        internal_tag = self._change_registry(tag, registry=INTERNAL_REGISTRY_HOST)

        for k, v in build_args.items():
            build_args_list.append(f'--build-arg="{k}={v}"')

        job = Job(
            {
                "metadata": {
                    "name": f"build-{job_id}",
                    "labels": {
                        "app.kubernetes.io/name": KUBERNETES_NAMESPACE,
                        "app.kubernetes.io/component": "builder",
                        "app.kubernetes.io/managed-by": "kr8s",
                    },
                },
                "spec": {
                    "backoffLimit": 0,
                    "ttlSecondsAfterFinished": JOB_COMPLETION_TTL,
                    "template": {
                        "spec": {
                            "restartPolicy": "Never",
                            "containers": [
                                {
                                    "name": "kaniko",
                                    "image": "gcr.io/kaniko-project/executor:latest",
                                    "args": [
                                        "--dockerfile=Dockerfile",
                                        "--context=dir:///workspace",
                                        f"--destination={internal_tag}",
                                        # Disabling --reproducible because it eats up a lot of CPU+RAM
                                        # https://github.com/GoogleContainerTools/kaniko/issues/1960
                                        # https://github.com/GoogleContainerTools/kaniko/pull/2477
                                        # "--reproducible",
                                        # cache args
                                        "--cache=true",
                                        "--cache-copy-layers",
                                        "--cache-run-layers",
                                        f"--cache-repo={INTERNAL_REGISTRY_HOST}/builder-cache",
                                        # outputs args
                                        "--digest-file=/dev/termination-log",
                                        # other kaniko conf
                                        f"--insecure-registry={INTERNAL_REGISTRY_HOST}",
                                        f"--skip-tls-verify-registry={INTERNAL_REGISTRY_HOST}",
                                        "--log-format=text",
                                        "--verbosity=info",
                                    ]
                                    + build_args_list,
                                    "volumeMounts": [
                                        {
                                            "name": "build-input",
                                            "mountPath": "/workspace",
                                        },
                                    ],
                                    "resources": {
                                        "requests": {
                                            "memory": "4Gi",
                                            "cpu": "2",
                                        },
                                        "limits": {
                                            "memory": "16Gi",
                                            "cpu": "4",
                                        },
                                        "ephemeral-storage": "10Gi",
                                    },
                                }
                            ],
                            "volumes": [
                                {
                                    "name": "build-input",
                                    "configMap": {
                                        "name": job_config.metadata.name,
                                    },
                                },
                            ],
                        }
                    },
                },
            }
        )

        return self._create_or_get(job)

    def _create_push_job(
        self,
        job_id: str,
        tag: str,
        push_secret: Secret,
    ) -> Job:
        internal_tag = self._change_registry(tag, registry=INTERNAL_REGISTRY_HOST)
        internal_reg, internal_repo, _ = parse_tag(internal_tag)

        run_cmds = [
            # push with credentials
            "echo Pushing image...",
            f"crane copy {internal_tag} {tag}",
            # cleanup image from internal registry
            "echo Cleaning up...",
            f"IMG_DIGEST=$(crane digest {internal_tag})",
            f"crane delete {internal_reg}/{internal_repo}@$IMG_DIGEST; echo Done",
        ]

        job = Job(
            {
                "metadata": {
                    # there should be only one push at a time, so keep this name unique to a push
                    "name": f"push-{job_id}",
                    "labels": {
                        "app.kubernetes.io/name": KUBERNETES_NAMESPACE,
                        "app.kubernetes.io/component": "builder",
                        "app.kubernetes.io/managed-by": "kr8s",
                    },
                },
                "spec": {
                    "backoffLimit": 0,
                    "ttlSecondsAfterFinished": JOB_COMPLETION_TTL,
                    "template": {
                        "spec": {
                            "restartPolicy": "Never",
                            "containers": [
                                {
                                    "name": "crane",
                                    # debug is needed for "sh" to be available
                                    "image": "gcr.io/go-containerregistry/crane:debug",
                                    "command": ["sh"],
                                    "args": ["-c", " && ".join(run_cmds)],
                                    "volumeMounts": [
                                        {
                                            "name": "push-secret",
                                            "mountPath": "/root/.docker/config.json",
                                            "subPath": "config.json",
                                            "readOnly": True,
                                        },
                                    ],
                                    "resources": {
                                        "requests": {
                                            "memory": "2Gi",
                                            "cpu": "1",
                                        },
                                        "limits": {
                                            "memory": "4Gi",
                                            "cpu": "2",
                                            "ephemeral-storage": "1Gi",
                                        },
                                    },
                                }
                            ],
                            "volumes": [
                                {
                                    "name": "push-secret",
                                    "secret": {
                                        "secretName": push_secret.metadata.name,
                                        "items": [
                                            {
                                                "key": ".dockerconfigjson",
                                                "path": "config.json",
                                            },
                                        ],
                                    },
                                },
                            ],
                        }
                    },
                },
            }
        )
        return self._create_or_get(job)

    def _create_push_secret(self, id: str, url: str, username: str, password: str):
        _secret = Secret(
            {
                "metadata": {
                    "name": f"push-secret-{id}",
                    "labels": {
                        "app.kubernetes.io/name": KUBERNETES_NAMESPACE,
                        "app.kubernetes.io/component": "builder",
                        "app.kubernetes.io/managed-by": "kr8s",
                    },
                },
                "type": "kubernetes.io/dockerconfigjson",
                "data": {
                    ".dockerconfigjson": create_dockerconfig_json(
                        registries=[
                            # authorize internal registry?
                            (INTERNAL_REGISTRY_HOST, "username", id),
                            (url, username, password),
                        ]
                    )
                },
            }
        )

        return self._create_or_get(_secret)

    def _create_or_get(self, obj: APIObject) -> APIObject:
        if not obj.exists():
            obj.create()
        else:
            obj.refresh()
        return obj
