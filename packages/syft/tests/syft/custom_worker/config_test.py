# stdlib
from itertools import chain
from itertools import combinations
import json
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from uuid import uuid4

# third party
from pydantic import BaseModel
import pytest
import yaml

# syft absolute
from syft.custom_worker.config import CustomWorkerConfig

# must follow the default values set in CustomBuildConfig class definition
DEFAULT_BUILD_CONFIG = {
    "gpu": False,
    "python_packages": [],
    "system_packages": [],
    "custom_cmds": [],
}


# must be set to the default value of CustomWorkerConfig.version
DEFAULT_WORKER_CONFIG_VERSION = "1"


CUSTOM_BUILD_CONFIG = {
    "gpu": True,
    "python_packages": ["toolz==0.12.0"],
    "system_packages": ["curl"],
    "custom_cmds": ["echo Hello"],
}


def generate_partial_custom_build_configs(
    full_config: Dict[str, Any]
) -> List[Dict[str, Any]]:
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
    build_config: Dict[str, Any], worker_config_version: Optional[str] = None
) -> Dict[str, Any]:
    worker_config = {"build": build_config}

    if worker_config_version is not None:
        worker_config["version"] = worker_config_version

    return worker_config


def get_full_build_config(build_config: Dict[str, Any]) -> Dict[str, Any]:
    return {**DEFAULT_BUILD_CONFIG, **build_config}


# in Pydantic v2 this would just be model.model_dump(mode='json')
def to_json_like_dict(model: BaseModel) -> dict:
    return json.loads(model.json())


@pytest.fixture
def worker_config(
    build_config: Dict[str, Any], worker_config_version: Optional[str]
) -> Dict[str, Any]:
    return get_worker_config(build_config, worker_config_version)


@pytest.fixture
def worker_config_yaml(tmp_path: Path, worker_config: Dict[str, Any]) -> Path:
    file_name = f"{uuid4().hex}.yaml"
    file_path = tmp_path / file_name
    with open(file_path, "w") as f:
        yaml.safe_dump(worker_config, f)

    yield file_path
    file_path.unlink()


@pytest.mark.parametrize("build_config", CUSTOM_BUILD_CONFIG_TEST_CASES)
@pytest.mark.parametrize("worker_config_version", ["2", None])
def test_load_custom_worker_config_file(
    build_config: Dict[str, Any],
    worker_config_version: Optional[str],
    worker_config_yaml: Path,
) -> None:
    parsed_worker_config_obj = CustomWorkerConfig.from_path(worker_config_yaml)

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