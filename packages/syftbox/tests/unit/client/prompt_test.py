import os
import tempfile
from pathlib import Path

import pytest

from syftbox.client.cli_setup import prompt_data_dir, prompt_email
from syftbox.lib.constants import DEFAULT_DATA_DIR
from syftbox.lib.validators import is_valid_dir, is_valid_email


@pytest.mark.parametrize(
    "path,expected",
    [
        ("/tmp", True),
        ("./test", True),
        (".", True),
        ("..", True),
        ("~", True),
        ("", False),  # Empty path = invalid
        ("/x", False),  # unwriteable path
    ],
)
def test_is_valid_dir(path, expected):
    """Test various email formats"""
    valid, reason = is_valid_dir(path, check_empty=False, check_writable=True)
    assert valid == expected, reason


def test_empty_dir():
    # Test with temp directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Valid empty directory
        valid, reason = is_valid_dir(temp_dir)
        assert valid
        assert reason == ""

        # Non-empty directory
        with open(os.path.join(temp_dir, "test.txt"), "w") as f:
            f.write("test")

        valid, reason = is_valid_dir(temp_dir)
        assert not valid
        assert "not empty" in reason.lower()


@pytest.mark.parametrize(
    "email,expected",
    [
        ("test@example.com", True),
        ("test.name@example.com", True),
        ("test+label@example.com", True),
        ("test@sub.example.com", True),
        ("a@b.c", True),
        ("", False),  # Empty email
        ("test@", False),  # Missing domain
        ("@example.com", False),  # Missing username
        ("test@example", False),  # Mising TLD
        ("test.example.com", False),  # Missing @
        ("test@@example.com", False),  # Double @
        ("test@exam ple.com", False),  # Space
        ("test@example..com", False),  # Double dots
    ],
)
def test_email_validation(email, expected):
    """Test various email formats"""
    assert is_valid_email(email) == expected


@pytest.mark.parametrize(
    "user_input,expected",
    [
        ("", Path(DEFAULT_DATA_DIR)),
        ("./valid/path", Path("./valid/path")),
    ],
)
def test_prompt_data_dir(user_input, expected, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *a, **k: user_input)
    monkeypatch.setattr("syftbox.client.cli_setup.is_valid_dir", lambda x: (True, ""))

    dir = prompt_data_dir()
    assert dir.absolute() == expected.absolute()


@pytest.mark.timeout(1)
def test_prompt_email(monkeypatch):
    valid_email = "test@example.com"

    monkeypatch.setattr("builtins.input", lambda *a, **k: valid_email)
    monkeypatch.setattr("syftbox.client.cli_setup.is_valid_dir", lambda x: (True, ""))

    email = prompt_email()
    assert email == valid_email
