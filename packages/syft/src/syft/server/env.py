# stdlib
import json
import subprocess  # nosec
import sys

# relative
from ..service.worker.utils import DEFAULT_WORKER_POOL_NAME
from ..types.uid import UID
from ..util.util import get_env
from ..util.util import str_to_bool

SERVER_PRIVATE_KEY = "SERVER_PRIVATE_KEY"
SERVER_UID = "SERVER_UID"
SERVER_TYPE = "SERVER_TYPE"
SERVER_NAME = "SERVER_NAME"
SERVER_SIDE_TYPE = "SERVER_SIDE_TYPE"

DEFAULT_ROOT_EMAIL = "DEFAULT_ROOT_EMAIL"
DEFAULT_ROOT_USERNAME = "DEFAULT_ROOT_USERNAME"
DEFAULT_ROOT_PASSWORD = "DEFAULT_ROOT_PASSWORD"  # nosec


def get_private_key_env() -> str | None:
    return get_env(SERVER_PRIVATE_KEY)


def get_server_type() -> str | None:
    return get_env(SERVER_TYPE, "datasite")


def get_server_name() -> str | None:
    return get_env(SERVER_NAME, None)


def get_server_side_type() -> str | None:
    return get_env(SERVER_SIDE_TYPE, "high")


def get_server_uid_env() -> str | None:
    return get_env(SERVER_UID)


def get_default_root_email() -> str | None:
    return get_env(DEFAULT_ROOT_EMAIL, "info@openmined.org")


def get_default_root_username() -> str | None:
    return get_env(DEFAULT_ROOT_USERNAME, "Jane Doe")


def get_default_root_password() -> str | None:
    return get_env(DEFAULT_ROOT_PASSWORD, "changethis")  # nosec


def get_enable_warnings() -> bool:
    return str_to_bool(get_env("ENABLE_WARNINGS", "False"))


def get_container_host() -> str | None:
    return get_env("CONTAINER_HOST")


def get_default_worker_image() -> str | None:
    return get_env("DEFAULT_WORKER_POOL_IMAGE")


def get_default_worker_pool_name() -> str | None:
    return get_env("DEFAULT_WORKER_POOL_NAME", DEFAULT_WORKER_POOL_NAME)


def get_default_bucket_name() -> str:
    env = get_env("DEFAULT_BUCKET_NAME")
    server_id = get_server_uid_env() or "syft-bucket"
    return env or server_id or "syft-bucket"


def get_default_worker_pool_pod_annotations() -> dict[str, str] | None:
    annotations = get_env("DEFAULT_WORKER_POOL_POD_ANNOTATIONS", "null")
    return json.loads(annotations)


def get_default_worker_pool_pod_labels() -> dict[str, str] | None:
    labels = get_env("DEFAULT_WORKER_POOL_POD_LABELS", "null")
    return json.loads(labels)


def in_kubernetes() -> bool:
    return get_container_host() == "k8s"


def get_venv_packages() -> str:
    try:
        # subprocess call is safe because it uses a fully qualified path and fixed arguments
        result = subprocess.run(
            [sys.executable, "-m", "pip", "list", "--format=freeze"],  # nosec
            capture_output=True,
            check=True,
            text=True,
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"An error occurred: {e.stderr}"


def get_syft_worker() -> bool:
    return str_to_bool(get_env("SYFT_WORKER", "false"))


def get_k8s_pod_name() -> str | None:
    return get_env("K8S_POD_NAME")


def get_syft_worker_uid() -> str | None:
    is_worker = get_syft_worker()
    pod_name = get_k8s_pod_name()
    uid = get_env("SYFT_WORKER_UID")
    # if uid is empty is a K8S worker, generate a uid from the pod name
    if (not uid) and is_worker and pod_name:
        uid = str(UID.with_seed(pod_name))
    return uid
