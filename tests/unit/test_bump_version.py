"""Check the version that bump_version.py writes into the pin of a dependent."""

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "bump_version.py"

TARGET = """\
[project]
name = "syft-thing"
version = "0.1.9"
dependencies = []
"""

DEPENDENT = """\
[project]
name = "syft-other"
version = "0.2.0"
dependencies = [
    "syft-thing==0.1.9",
]

[tool.uv.sources]
"syft-thing" = { workspace = true }
"""


@pytest.fixture
def fake_repo(tmp_path):
    (tmp_path / "packages" / "syft-thing").mkdir(parents=True)
    (tmp_path / "packages" / "syft-other").mkdir(parents=True)
    (tmp_path / "packages" / "syft-thing" / "pyproject.toml").write_text(TARGET)
    (tmp_path / "packages" / "syft-other" / "pyproject.toml").write_text(DEPENDENT)
    return tmp_path


def _run(fake_repo, *args):
    spec = importlib.util.spec_from_file_location("bump_version_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.REPO_ROOT = fake_repo
    argv = [str(SCRIPT), "syft-thing", "patch", *args]
    old = sys.argv
    sys.argv = argv
    try:
        module.main()
    finally:
        sys.argv = old


def _versions(fake_repo):
    target = (fake_repo / "packages" / "syft-thing" / "pyproject.toml").read_text()
    dependent = (fake_repo / "packages" / "syft-other" / "pyproject.toml").read_text()
    source = next(
        line for line in target.splitlines() if line.startswith("version")
    ).split('"')[1]
    pin = next(line for line in dependent.splitlines() if "syft-thing==" in line)
    return source, pin.split("==")[1].split('"')[0]


def test_default_pins_dependents_to_the_bumped_version(fake_repo):
    _run(fake_repo)
    source, pin = _versions(fake_repo)
    assert source == "0.1.10"
    assert pin == "0.1.10"


def test_published_pins_dependents_to_the_version_just_released(fake_repo):
    # A release publishes the version on the branch, then bumps the version. The
    # monorepo releases a dependent later in the same run. The pin must therefore
    # name a version that PyPI already has.
    _run(fake_repo, "--dependents", "published")
    source, pin = _versions(fake_repo)
    assert source == "0.1.10"
    assert pin == "0.1.9"


def test_dependent_pin_is_a_published_version_for_every_release_order(fake_repo):
    # This test covers the monorepo order. syft-perms releases before syft-job. If
    # the script pins a dependent to the new version, syft-job publishes a
    # dependency that PyPI does not have.
    _run(fake_repo, "--dependents", "published")
    _, pin = _versions(fake_repo)
    assert pin == "0.1.9", "a dependent must pin the version that the release published"
