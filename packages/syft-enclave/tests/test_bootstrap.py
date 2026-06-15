"""Tests for the token bootstrap dispatcher and providers."""

from __future__ import annotations

import os
import stat

import pytest

from syft_enclaves import bootstrap


@pytest.fixture
def clean_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith("SYFT_"):
            monkeypatch.delenv(key, raising=False)
    return monkeypatch


def test_envvar_writes_token_with_correct_perms(clean_env, tmp_path):
    token_path = tmp_path / "token.json"
    clean_env.setenv("SYFT_ENCLAVE_TOKEN_PATH", str(token_path))
    clean_env.setenv("SYFT_BOOTSTRAP", "envvar")
    clean_env.setenv("SYFT_ENCLAVE_TOKEN_CONTENT", '{"refresh_token": "abc"}')

    bootstrap.run()

    assert token_path.read_bytes() == b'{"refresh_token": "abc"}'
    assert stat.S_IMODE(token_path.stat().st_mode) == 0o600


def test_file_fallback_accepts_existing_token(clean_env, tmp_path):
    token_path = tmp_path / "token.json"
    token_path.write_bytes(b"existing")
    clean_env.setenv("SYFT_ENCLAVE_TOKEN_PATH", str(token_path))

    bootstrap.run()  # SYFT_BOOTSTRAP unset, file exists → no-op

    assert token_path.read_bytes() == b"existing"


def test_no_bootstrap_no_file_exits(clean_env, tmp_path):
    clean_env.setenv("SYFT_ENCLAVE_TOKEN_PATH", str(tmp_path / "token.json"))
    with pytest.raises(SystemExit):
        bootstrap.run()


def test_unknown_provider_exits(clean_env, tmp_path):
    clean_env.setenv("SYFT_ENCLAVE_TOKEN_PATH", str(tmp_path / "token.json"))
    clean_env.setenv("SYFT_BOOTSTRAP", "frobnitz")
    with pytest.raises(SystemExit):
        bootstrap.run()


def test_envvar_missing_content_raises(clean_env):
    clean_env.setenv("SYFT_BOOTSTRAP", "envvar")
    with pytest.raises(RuntimeError, match="SYFT_ENCLAVE_TOKEN_CONTENT"):
        bootstrap.envvar_provider()


def test_wif_pipeline_wiring(clean_env, tmp_path, monkeypatch):
    """Stub the three HTTP-ish calls and verify the data flows through."""
    monkeypatch.setattr(bootstrap, "request_attestation_jwt", lambda aud: "the-jwt")
    monkeypatch.setattr(bootstrap, "sts_token_exchange", lambda jwt, aud: "fed-tok")
    monkeypatch.setattr(
        bootstrap, "secret_manager_access", lambda res, bearer: b"secret-payload"
    )

    token_path = tmp_path / "token.json"
    clean_env.setenv("SYFT_ENCLAVE_TOKEN_PATH", str(token_path))
    clean_env.setenv("SYFT_BOOTSTRAP", "wif")
    clean_env.setenv("SYFT_BOOTSTRAP_WIF_AUDIENCE", "//iam.googleapis.com/aud")
    clean_env.setenv(
        "SYFT_BOOTSTRAP_WIF_SECRET",
        "projects/123/secrets/x/versions/latest",
    )

    bootstrap.run()

    assert token_path.read_bytes() == b"secret-payload"
    assert stat.S_IMODE(token_path.stat().st_mode) == 0o600


def test_wif_missing_env_raises(clean_env):
    with pytest.raises(RuntimeError, match="SYFT_BOOTSTRAP_WIF_AUDIENCE"):
        bootstrap.wif_provider()


def test_sa_pipeline_wiring(clean_env, tmp_path, monkeypatch):
    """Stub the metadata server + Secret Manager call and verify data flows through."""

    class _FakeMetaResponse:
        status_code = 200
        text = ""

        def json(self):
            return {"access_token": "sa-access-tok"}

    monkeypatch.setattr(bootstrap.requests, "get", lambda *a, **kw: _FakeMetaResponse())
    monkeypatch.setattr(
        bootstrap, "secret_manager_access", lambda res, bearer: b"sa-secret-payload"
    )

    token_path = tmp_path / "token.json"
    clean_env.setenv("SYFT_ENCLAVE_TOKEN_PATH", str(token_path))
    clean_env.setenv("SYFT_BOOTSTRAP", "sa")
    clean_env.setenv(
        "SYFT_BOOTSTRAP_SA_SECRET",
        "projects/123/secrets/x/versions/latest",
    )

    bootstrap.run()

    assert token_path.read_bytes() == b"sa-secret-payload"
    assert stat.S_IMODE(token_path.stat().st_mode) == 0o600


def test_sa_missing_env_raises(clean_env):
    with pytest.raises(RuntimeError, match="SYFT_BOOTSTRAP_SA_SECRET"):
        bootstrap.sa_provider()
