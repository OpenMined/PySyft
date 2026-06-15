import random
import sys
from pathlib import Path

import pytest

import syft_client as sc
from syft_client.sync.syftbox_manager import SyftboxManager


def _create_tmp_code_folder() -> Path:
    """Create a temp folder with a top-level and a nested Python file."""
    tmp_dir = Path("/tmp/syft-datasets-code-testing") / str(
        random.randint(1, 1_000_000)
    )
    (tmp_dir / "utils").mkdir(parents=True, exist_ok=True)
    (tmp_dir / "helpers.py").write_text(
        "def double(x):\n    return x * 2\n\nVALUE = 'top-level'\n"
    )
    (tmp_dir / "utils" / "nested.py").write_text(
        "def triple(x):\n    return x * 3\n\nVALUE = 'nested'\n"
    )
    return tmp_dir


def _create_tmp_private_folder() -> Path:
    tmp_dir = Path("/tmp/syft-datasets-code-testing") / (
        "private_" + str(random.randint(1, 1_000_000))
    )
    tmp_dir.mkdir(parents=True, exist_ok=True)
    (tmp_dir / "private.txt").write_text("private payload")
    return tmp_dir


def _create_local_dataset(do_manager, name: str) -> None:
    """Create a dataset on the DO without going through the upload/sync layer.

    `SyftboxManager.create_dataset` uploads files into a collection that
    flattens nested folder structure, which is unrelated to what we want
    to test here. Calling `dataset_manager.create` directly populates the
    local mock_dir with the original (nested) layout.
    """
    do_manager.dataset_manager.create(
        name=name,
        mock_path=_create_tmp_code_folder(),
        private_path=_create_tmp_private_folder(),
        summary="Code dataset",
    )


def test_load_dataset_code_top_level_and_nested():
    _, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        use_in_memory_cache=False,
    )
    _create_local_dataset(do_manager, "my_code_dataset")

    top = sc.load_dataset_code("my_code_dataset.helpers", client=do_manager)
    assert top.double(4) == 8
    assert top.VALUE == "top-level"

    nested = sc.load_dataset_code("my_code_dataset.utils.nested", client=do_manager)
    assert nested.triple(4) == 12
    assert nested.VALUE == "nested"

    # Default module name is the leaf segment.
    assert sys.modules.get("helpers") is top
    assert sys.modules.get("nested") is nested


def test_load_dataset_code_explicit_module_name():
    _, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        use_in_memory_cache=False,
    )
    _create_local_dataset(do_manager, "my_code_dataset")

    mod = sc.load_dataset_code(
        "my_code_dataset.helpers",
        client=do_manager,
        module_name="my_custom_name",
    )
    assert sys.modules.get("my_custom_name") is mod


def test_load_dataset_code_invalid_path_raises():
    _, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        use_in_memory_cache=False,
    )
    with pytest.raises(ValueError, match="Invalid path"):
        sc.load_dataset_code("just_a_name", client=do_manager)


def test_load_dataset_code_missing_file_raises():
    _, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        use_in_memory_cache=False,
    )
    _create_local_dataset(do_manager, "my_code_dataset")

    with pytest.raises(FileNotFoundError, match="Code file not found"):
        sc.load_dataset_code("my_code_dataset.does_not_exist", client=do_manager)
