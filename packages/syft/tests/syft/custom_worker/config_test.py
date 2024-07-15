# stdlib
from itertools import chain
from itertools import combinations
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

# third party
from pydantic import BaseModel
import pytest
import yaml

# syft absolute
import syft as sy
from syft.custom_worker.config import CustomBuildConfig
from syft.custom_worker.config import CustomWorkerConfig
from syft.custom_worker.config import DockerWorkerConfig


# in Pydantic v2 this would just be model.model_dump(mode='json')
def to_json_like_dict(model: BaseModel) -> dict[str, Any]:
    return json.loads(model.json())


DEFAULT_BUILD_CONFIG = {
    "gpu": False,
    "python_packages": [],
    "system_packages": [],
    "custom_cmds": [],
}
# must follow the default values set in CustomBuildConfig class definition
assert DEFAULT_BUILD_CONFIG == to_json_like_dict(CustomBuildConfig())


DEFAULT_WORKER_CONFIG_VERSION = "1"
# must be set to the default value of CustomWorkerConfig.version
assert (
    DEFAULT_WORKER_CONFIG_VERSION
    == CustomWorkerConfig(build=CustomBuildConfig()).version
)


CUSTOM_BUILD_CONFIG = {
    "gpu": True,
    "python_packages": ["toolz==0.12.0"],
    "system_packages": ["curl"],
    "custom_cmds": ["echo Hello"],
}


def generate_partial_custom_build_configs(
    full_config: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    generate_partial_custom_build_configs({
        "gpu": True,
        "python_packages": ["toolz==0.12.0"],
        "system_packages": ["curl"],
        "custom_cmds": ["echo Hello"],
    })

    would return

    [
        {},
        {"gpu": True},
        {"python_packages": ["toolz==0.12.0"]},
        ...
        {"gpu": True, "python_packages": ["toolz==0.12.0"]},
        ...,
        {
            "gpu": True,
            "python_packages": ["toolz==0.12.0"],
            "system_packages": ["curl"],
            "custom_cmds": ["echo Hello"],
        }
    ]
    """
    config_kvs = list(full_config.items())

    return [
        dict(kvs)
        for kvs in chain.from_iterable(
            combinations(config_kvs, i) for i in range(len(config_kvs) + 1)
        )
    ]


CUSTOM_BUILD_CONFIG_TEST_CASES = generate_partial_custom_build_configs(
    CUSTOM_BUILD_CONFIG
)


def get_worker_config(
    build_config: dict[str, Any], worker_config_version: str | None = None
) -> dict[str, Any]:
    worker_config = {"build": build_config}

    if worker_config_version is not None:
        worker_config["version"] = worker_config_version

    return worker_config


def get_full_build_config(build_config: dict[str, Any]) -> dict[str, Any]:
    return {**DEFAULT_BUILD_CONFIG, **build_config}


@pytest.fixture
def worker_config(
    build_config: dict[str, Any], worker_config_version: str | None
) -> dict[str, Any]:
    yield get_worker_config(build_config, worker_config_version)


@pytest.fixture
def worker_config_yaml(tmp_path: Path, worker_config: dict[str, Any]) -> Path:
    file_name = f"{uuid4().hex}.yaml"
    file_path = tmp_path / file_name
    with open(file_path, "w") as f:
        yaml.safe_dump(worker_config, f)

    yield file_path
    file_path.unlink()


METHODS = ["from_dict", "from_str", "from_path"]


@pytest.mark.parametrize("build_config", CUSTOM_BUILD_CONFIG_TEST_CASES)
@pytest.mark.parametrize("worker_config_version", ["2", None])
@pytest.mark.parametrize("method", METHODS)
def test_load_custom_worker_config(
    build_config: dict[str, Any],
    worker_config_version: str | None,
    worker_config_yaml: Path,
    method: str,
) -> None:
    if method == "from_path":
        parsed_worker_config_obj = CustomWorkerConfig.from_path(worker_config_yaml)
    elif method == "from_str":
        parsed_worker_config_obj = CustomWorkerConfig.from_str(
            worker_config_yaml.read_text()
        )
    elif method == "from_dict":
        with open(worker_config_yaml) as f:
            config = yaml.safe_load(f)
        parsed_worker_config_obj = CustomWorkerConfig.from_dict(config)
    else:
        raise ValueError(f"method must be one of {METHODS}")

    worker_config_version = (
        DEFAULT_WORKER_CONFIG_VERSION
        if worker_config_version is None
        else worker_config_version
    )

    expected = get_worker_config(
        build_config=get_full_build_config(build_config),
        worker_config_version=worker_config_version,
    )

    assert to_json_like_dict(parsed_worker_config_obj) == expected


DOCKER_METHODS = ["from_str", "from_path"]
DOCKER_CONFIG_OPENDP = f"""
    FROM openmined/syft-backend:{sy.__version__}
    RUN pip install opendp
"""


@pytest.fixture
def dockerfile_path(tmp_path: Path) -> Path:
    file_name = f"{uuid4().hex}.Dockerfile"
    file_path = tmp_path / file_name

    with open(file_path, "w") as f:
        f.write(DOCKER_CONFIG_OPENDP)

    yield file_path
    file_path.unlink()


@pytest.mark.parametrize("method", DOCKER_METHODS)
def test_docker_worker_config(dockerfile_path: Path, method: str) -> None:
    description = "I want to do some cool DS stuff with Syft and OpenDP"
    if method == "from_str":
        docker_config = DockerWorkerConfig(
            dockerfile=dockerfile_path.read_text(), description=description
        )
    elif method == "from_path":
        docker_config = DockerWorkerConfig.from_path(
            path=dockerfile_path, description=description
        )
    else:
        raise ValueError(f"method must be one of {METHODS}")

    assert docker_config.dockerfile == dockerfile_path.read_text().strip()
    assert docker_config.description == description
    new_description = description + f" (syft version is {sy.__version__})"
    docker_config.set_description(description_text=new_description)
    assert docker_config.description == new_description
