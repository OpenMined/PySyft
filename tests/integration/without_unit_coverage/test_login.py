"""
Integration tests for login_ds and login_do functions.

These tests verify that the actual login flow works correctly in CI,
catching issues that would otherwise only be discovered when running notebooks.
"""

import os
from pathlib import Path

import pytest

import syft_client as sc
from syft_client.sync.syftbox_manager import SyftboxManager


SYFT_CLIENT_DIR = Path(__file__).parent.parent.parent.parent
CREDENTIALS_DIR = SYFT_CLIENT_DIR / "credentials"

# Credentials from CI environment
FILE_DO = os.environ.get("ai_audit_credentials_fname_do", "token_do.json")
EMAIL_DO = os.environ.get("AI_AUDIT_EMAIL_DO")

FILE_DS = os.environ.get("ai_audit_credentials_fname_ds", "token_ds.json")
EMAIL_DS = os.environ.get("AI_AUDIT_EMAIL_DS")

token_path_do = CREDENTIALS_DIR / FILE_DO
token_path_ds = CREDENTIALS_DIR / FILE_DS


@pytest.fixture
def check_credentials():
    """Verify credentials exist before running tests."""
    if not EMAIL_DO or not EMAIL_DS:
        pytest.skip(
            "AI_AUDIT_EMAIL_DO and AI_AUDIT_EMAIL_DS environment variables required"
        )
    if not token_path_do.exists() or not token_path_ds.exists():
        pytest.skip(
            f"Token files not found. Expected: {token_path_do} and {token_path_ds}"
        )
    yield


@pytest.mark.flaky(reruns=3, reruns_delay=2)
@pytest.mark.usefixtures("check_credentials")
def test_login_ds():
    """Test that login_ds works with Jupyter environment and token credentials.

    This tests the actual code path that notebook users hit when calling sc.login_ds().
    """
    client = sc.login_ds(
        email=EMAIL_DS,
        token_path=token_path_ds,
        sync=False,  # Don't sync to avoid side effects in CI
        load_peers=False,
    )

    assert client is not None
    assert isinstance(client, SyftboxManager)
    assert client.email == EMAIL_DS


@pytest.mark.flaky(reruns=3, reruns_delay=2)
@pytest.mark.usefixtures("check_credentials")
def test_login_do():
    """Test that login_do works with Jupyter environment and token credentials.

    This tests the actual code path that notebook users hit when calling sc.login_do().
    """
    client = sc.login_do(
        email=EMAIL_DO,
        token_path=token_path_do,
        sync=False,  # Don't sync to avoid side effects in CI
        load_peers=False,
    )

    assert client is not None
    assert isinstance(client, SyftboxManager)
    assert client.email == EMAIL_DO


@pytest.mark.flaky(reruns=3, reruns_delay=2)
@pytest.mark.usefixtures("check_credentials")
def test_login_ds_with_sync_and_load_peers():
    """Test that login_ds works with sync and load_peers enabled."""
    client = sc.login_ds(
        email=EMAIL_DS,
        token_path=token_path_ds,
        sync=True,
        load_peers=True,
    )

    assert client is not None
    assert isinstance(client, SyftboxManager)
    assert client.email == EMAIL_DS
    # Verify client has expected attributes
    assert hasattr(client, "datasets")
    assert hasattr(client, "job_client")


@pytest.mark.flaky(reruns=3, reruns_delay=2)
@pytest.mark.usefixtures("check_credentials")
def test_login_do_with_sync_and_load_peers():
    """Test that login_do works with sync and load_peers enabled."""
    client = sc.login_do(
        email=EMAIL_DO,
        token_path=token_path_do,
        sync=True,
        load_peers=True,
    )

    assert client is not None
    assert isinstance(client, SyftboxManager)
    assert client.email == EMAIL_DO
    # Verify client has expected attributes
    assert hasattr(client, "datasets")
    assert hasattr(client, "job_client")


@pytest.mark.flaky(reruns=3, reruns_delay=2)
@pytest.mark.usefixtures("check_credentials")
def test_login_ds_email_mismatch_raises():
    """Login must fail fast when the email doesn't match the token's account."""
    with pytest.raises(ValueError, match="Token/email mismatch") as exc_info:
        sc.login_ds(
            email=EMAIL_DO,  # intentionally wrong: DO's email with DS's token
            token_path=token_path_ds,
            sync=False,
            load_peers=False,
        )

    msg = str(exc_info.value)
    assert EMAIL_DO in msg
    assert EMAIL_DS in msg


@pytest.mark.flaky(reruns=3, reruns_delay=2)
def test_login_ds_missing_token_path_raises(monkeypatch):
    """Test that login_ds raises error when token_path is missing in Jupyter env."""
    # Unset the env var so the test can verify the error is raised
    monkeypatch.delenv("SYFTCLIENT_TOKEN_PATH", raising=False)
    with pytest.raises(NotImplementedError, match="token path"):
        sc.login_ds(
            email="test@test.com",
            token_path=None,
            sync=False,
            load_peers=False,
        )


@pytest.mark.flaky(reruns=3, reruns_delay=2)
def test_login_do_missing_token_path_raises(monkeypatch):
    """Test that login_do raises error when token_path is missing in Jupyter env."""
    # Unset the env var so the test can verify the error is raised
    monkeypatch.delenv("SYFTCLIENT_TOKEN_PATH", raising=False)
    with pytest.raises(NotImplementedError, match="token path"):
        sc.login_do(
            email="test@test.com",
            token_path=None,
            sync=False,
            load_peers=False,
        )
