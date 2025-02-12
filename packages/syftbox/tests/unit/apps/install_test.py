from pathlib import Path
from subprocess import CalledProcessError

import pytest

from syftbox.app.install import clone_repository, sanitize_git_path


def test_valid_git_path():
    path = "Example/Repository"
    output_path = sanitize_git_path(path)
    assert path == output_path


def test_valid_git_url():
    path = "Example/Repository"
    http_url = f"http://github.com/{path}"
    output_path = sanitize_git_path(http_url)
    assert path == output_path

    https_url = f"https://github.com/{path}"
    output_path = sanitize_git_path(https_url)
    assert path == output_path


def test_invalid_git_path():
    path = "..Example/../Repository"
    with pytest.raises(ValueError) as excpt:
        _ = sanitize_git_path(path)
        assert excpt.value == "Invalid Git repository path format."


def test_second_invalid_git_path():
    path = "http://example.com"
    with pytest.raises(ValueError) as excpt:
        _ = sanitize_git_path(path)
        assert excpt.value == "Invalid Git repository path format."


def test_clone_valid_repository(monkeypatch):
    count = 0

    def mock_subproc_run(*args, **kwargs):
        nonlocal count
        count += 1
        return 0

    monkeypatch.setattr("subprocess.run", mock_subproc_run)

    path = "OpenMined/logged_in"
    temp_path = clone_repository(path, "main")
    assert count == 3
    assert isinstance(temp_path, Path)


def test_clone_repository_to_an_existent_path(monkeypatch):
    count = 0

    def mock_subproc_run(*args, **kwargs):
        nonlocal count
        count += 1
        return 0

    monkeypatch.setattr("subprocess.run", mock_subproc_run)

    # First call will make the repository path exist
    path = "OpenMined/logged_in"
    temp_path = clone_repository(path, "main")
    assert isinstance(temp_path, Path)
    assert count == 3

    # Second call must clone it again without any exception (replaces the old one).
    temp_path = clone_repository(path, "main")
    assert isinstance(temp_path, Path)
    assert count == 6


def test_clone_invalid_repository(monkeypatch):
    count = 0

    def mock_subproc_run(*args, **kwargs):
        nonlocal count
        count += 1
        cmd = args[0]

        if cmd[0] == "git" and cmd[1] == "ls-remote" and "Invalid" in cmd[2]:
            raise CalledProcessError(1, cmd)

        return 0

    monkeypatch.setattr("subprocess.run", mock_subproc_run)

    path = "InvalidUser/InvalidRepo"
    with pytest.raises(ValueError) as excpt:
        _ = clone_repository(path, "main")
        assert "Cannot access repository" in excpt.value
