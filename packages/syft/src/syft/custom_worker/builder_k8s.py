# stdlib
from hashlib import sha256
import os
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional

# third party
import kr8s
from kr8s.objects import APIObject
from kr8s.objects import ConfigMap
from kr8s.objects import Job

# relative
from .builder_types import BuilderBase
from .builder_types import ImageBuildResult
from .builder_types import ImagePushResult
from .k8s import BUILD_OUTPUT_PVC
from .k8s import JOB_COMPLETION_TTL
from .k8s import KUBERNETES_NAMESPACE

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
        job_id = self._new_job_id(tag)

        if dockerfile:
            pass
        elif dockerfile_path:
            dockerfile = dockerfile_path.read_text()

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

        try:
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
            config.delete()

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
        # Create and start the job
        job_id = self._new_job_id(tag)
        job = self._create_push_job(
            job_id=job_id,
            tag=tag,
            username=username,
            password=password,
            registry_url=registry_url,
        )
        job.wait(["condition=Complete", "condition=Failed"])
        exit_code = self._get_container_exit_code(job)[0]
        return ImagePushResult(logs=self._get_logs(job), exit_code=exit_code)

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

    def _check_success(self, job: Job) -> bool:
        # TODO
        return True

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

    def _create_kaniko_build_job(
        self,
        job_id: str,
        tag: str,
        job_config: ConfigMap,
        build_args: Optional[Dict] = None,
    ) -> Job:
        # for push
        build_args = build_args or {}
        tag_hash = self._get_tag_hash(tag)
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
                                        # build_args
                                        "--dockerfile=Dockerfile",
                                        "--context=dir:///workspace",
                                        f"--destination={tag}",
                                        "--reproducible",
                                        # Build outputs
                                        f"--tar-path=/output/{tag_hash}.tar",
                                        "--digest-file=/dev/termination-log",
                                        "--no-push",
                                        # other kaniko conf
                                        "--log-format=text",
                                        "--verbosity=info",
                                    ],
                                    "volumeMounts": [
                                        {
                                            "name": "build-input",
                                            "mountPath": "/workspace",
                                        },
                                        {
                                            "name": "build-output",
                                            "mountPath": "/output",
                                            "readOnly": False,
                                        },
                                    ],
                                }
                            ],
                            "volumes": [
                                {
                                    "name": "build-input",
                                    "configMap": {
                                        "name": job_config.metadata.name,
                                    },
                                },
                                {
                                    "name": "build-output",
                                    "persistentVolumeClaim": {
                                        "claimName": BUILD_OUTPUT_PVC,
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
        username: str,
        password: str,
        registry_url: Optional[str] = None,
    ) -> Job:
        tag_hash = self._get_tag_hash(tag)
        registry_url = registry_url or tag.split("/")[0]

        extra_flags = ""
        if os.getenv("DEV_MODE") == "True":
            extra_flags = "--insecure"

        run_cmds = [
            "echo Logging in to $REG_URL with user $REG_USERNAME...",
            # login to registry
            "crane auth login $REG_URL -u $REG_USERNAME -p $REG_PASSWORD",
            # push with credentials
            "echo Pushing image....",
            f"crane push --image-refs /dev/termination-log {extra_flags} /output/{tag_hash}.tar {tag}",
            # cleanup built tarfile
            "echo Cleaning up tar....",
            f"rm /output/{tag_hash}.tar",
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
                                    "env": [
                                        {
                                            "name": "REG_URL",
                                            "value": registry_url,
                                        },
                                        {
                                            "name": "REG_USERNAME",
                                            "value": username,
                                        },
                                        {
                                            "name": "REG_PASSWORD",
                                            "value": password,
                                        },
                                    ],
                                    "command": ["sh"],
                                    "args": ["-c", " && ".join(run_cmds)],
                                    "volumeMounts": [
                                        {
                                            "name": "build-output",
                                            "mountPath": "/output",
                                            "readOnly": False,
                                        },
                                    ],
                                }
                            ],
                            "volumes": [
                                {
                                    "name": "build-output",
                                    "persistentVolumeClaim": {
                                        "claimName": BUILD_OUTPUT_PVC,
                                    },
                                },
                            ],
                        }
                    },
                },
            }
        )
        return self._create_or_get(job)

    def _create_or_get(self, obj: APIObject) -> APIObject:
        if not obj.exists():
            obj.create()
        else:
            obj.refresh()
        return obj
