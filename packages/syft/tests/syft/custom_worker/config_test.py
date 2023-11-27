# stdlib
import json
from pathlib import Path
from uuid import uuid4

# third party
from pydantic import BaseModel
import pytest
import yaml

# syft absolute
from syft.custom_worker.config import CustomWorkerConfig

CUSTOM_BUILD_CONFIG = {
    "gpu": True,
    "python_packages": ["toolz==0.12.0"],
    "system_packages": ["curl"],
    "custom_cmds": [],
}


CUSTOM_WORKER_CONFIG = {
    "build": CUSTOM_BUILD_CONFIG,
    "version": "1",
}


# in Pydantic v2 this would just be model.model_dump(mode='json')
def to_json_like_dict(model: BaseModel) -> dict:
    return json.loads(model.json())


@pytest.fixture
def worker_config_yaml(tmp_path: Path) -> Path:
    file_name = f"{uuid4().hex}.yaml"
    file_path = tmp_path / file_name
    with open(file_path, "w") as f:
        yaml.safe_dump(CUSTOM_WORKER_CONFIG, f)

    return file_path


def test_load_custom_worker_config_file(worker_config_yaml: Path) -> None:
    worker_config = CustomWorkerConfig.from_path(worker_config_yaml)
    assert to_json_like_dict(worker_config) == CUSTOM_WORKER_CONFIG
