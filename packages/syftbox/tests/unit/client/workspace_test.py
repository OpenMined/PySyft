from pathlib import Path

import pytest

from syftbox.lib.workspace import SyftWorkspace


@pytest.fixture
def temp_workspace(tmp_path):
    """Fixture to create a temporary workspace directory."""
    return SyftWorkspace(tmp_path / "workspace")


def test_workspace_init():
    # Test with string path
    workspace = SyftWorkspace("/tmp/test")
    assert isinstance(workspace.data_dir, Path)

    # Test with Path object
    path_obj = Path("~/test2")
    workspace = SyftWorkspace(path_obj)

    assert isinstance(workspace.data_dir, Path)
    assert workspace.data_dir.is_absolute()
    assert "~" not in str(workspace.data_dir)


def test_workspace_directory_structure(tmp_path):
    workspace = SyftWorkspace(tmp_path)

    assert not workspace.datasites.exists()
    assert not workspace.plugins.exists()
    assert not workspace.apps.exists()

    # Create directories
    workspace.mkdirs()

    # Verify directory structure
    assert workspace.data_dir.is_dir()
    assert workspace.datasites.is_dir()
    assert workspace.plugins.is_dir()
    assert workspace.apps.is_dir()
