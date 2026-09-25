"""Shared test fixtures for syft-bg tests."""

import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_config(temp_dir):
    """Create a sample config file."""
    config_path = temp_dir / "config.yaml"
    config_path.write_text("""
do_email: test@example.com
syftbox_root: /tmp/syftbox

notify:
  interval: 30
  monitor_jobs: true
  monitor_peers: true

approve:
  interval: 5
  auto_approvals:
    enabled: true
  peers:
    enabled: false
    approved_domains: []
""")
    return config_path


@pytest.fixture
def sample_script(temp_dir):
    """Create a sample script file for hash testing."""
    script_path = temp_dir / "main.py"
    script_path.write_text('print("hello world")\n')
    return script_path
