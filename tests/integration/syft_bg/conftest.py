"""Shared test fixtures for syft-bg integration tests."""

import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_script(temp_dir):
    """Create a sample script file for hash testing."""
    script_path = temp_dir / "main.py"
    script_path.write_text('print("hello world")\n')
    return script_path


@pytest.fixture
def sample_scripts(temp_dir):
    """Create multiple sample script files."""
    main = temp_dir / "main.py"
    main.write_text('print("hello")\n')
    utils = temp_dir / "utils.py"
    utils.write_text("def helper(): pass\n")
    return [main, utils]
