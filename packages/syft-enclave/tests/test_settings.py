"""Tests for EnclaveSettings — environment-driven runner configuration."""

import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from syft_enclaves import EnclaveSettings


@pytest.fixture
def clean_env(monkeypatch):
    """Strip any inherited SYFT_ENCLAVE_* vars so each test starts clean."""
    for key in list(os.environ):
        if key.startswith("SYFT_ENCLAVE_"):
            monkeypatch.delenv(key, raising=False)
    return monkeypatch


@pytest.fixture
def required_env(clean_env):
    """Set the minimal required SYFT_ENCLAVE_* vars (email + data_owners + token_path)."""
    clean_env.setenv("SYFT_ENCLAVE_EMAIL", "enclave@openmined.org")
    clean_env.setenv("SYFT_ENCLAVE_DATA_OWNERS", "do1@openmined.org,do2@openmined.org")
    clean_env.setenv("SYFT_ENCLAVE_TOKEN_PATH", "/secrets/token.json")
    return clean_env


def test_defaults_applied_when_required_fields_set(required_env):
    settings = EnclaveSettings(_env_file=None)

    assert settings.email == "enclave@openmined.org"
    assert settings.data_owners == ["do1@openmined.org", "do2@openmined.org"]
    assert settings.token_path == Path("/secrets/token.json")
    assert settings.poll_interval == 1
    assert settings.require_tee is False
    assert settings.log_level == "INFO"
    assert settings.fresh_state is True  # default: always start with a clean slate
    assert settings.use_encryption is True  # default: encryption on


def test_data_owners_parsed_from_comma_separated_string(clean_env):
    clean_env.setenv("SYFT_ENCLAVE_EMAIL", "enclave@openmined.org")
    clean_env.setenv("SYFT_ENCLAVE_DATA_OWNERS", " do1@x.com , do2@y.com ")
    settings = EnclaveSettings(_env_file=None)
    assert settings.data_owners == ["do1@x.com", "do2@y.com"]


def test_missing_data_owners_raises(clean_env):
    clean_env.setenv("SYFT_ENCLAVE_EMAIL", "enclave@openmined.org")
    clean_env.setenv("SYFT_ENCLAVE_TOKEN_PATH", "/secrets/token.json")
    with pytest.raises(ValidationError, match="data_owners"):
        EnclaveSettings(_env_file=None)


def test_missing_email_raises(clean_env):
    clean_env.setenv("SYFT_ENCLAVE_TOKEN_PATH", "/secrets/token.json")
    with pytest.raises(ValidationError, match="email"):
        EnclaveSettings(_env_file=None)


def test_token_path_defaults_to_entrypoint_write_location(clean_env):
    # The default must match the path docker/entrypoint.sh writes to.
    clean_env.setenv("SYFT_ENCLAVE_EMAIL", "enclave@openmined.org")
    clean_env.setenv("SYFT_ENCLAVE_DATA_OWNERS", "do1@openmined.org")
    settings = EnclaveSettings(_env_file=None)
    assert settings.token_path == Path("/run/syft-enclave/token.json")


def test_env_prefix_and_type_coercion(required_env):
    required_env.setenv("SYFT_ENCLAVE_POLL_INTERVAL", "5")
    required_env.setenv("SYFT_ENCLAVE_REQUIRE_TEE", "true")
    required_env.setenv("SYFT_ENCLAVE_LOG_LEVEL", "DEBUG")

    settings = EnclaveSettings(_env_file=None)

    assert settings.token_path == Path("/secrets/token.json")
    assert settings.poll_interval == 5
    assert settings.require_tee is True
    assert settings.log_level == "DEBUG"


def test_invalid_log_level_raises(required_env):
    required_env.setenv("SYFT_ENCLAVE_LOG_LEVEL", "TRACE")

    with pytest.raises(ValidationError, match="log_level"):
        EnclaveSettings(_env_file=None)


def test_poll_interval_must_be_positive(required_env):
    required_env.setenv("SYFT_ENCLAVE_POLL_INTERVAL", "0")

    with pytest.raises(ValidationError, match="poll_interval"):
        EnclaveSettings(_env_file=None)


def test_settings_are_frozen(required_env):
    settings = EnclaveSettings(_env_file=None)

    with pytest.raises(ValidationError):
        settings.poll_interval = 99


def test_fresh_state_can_be_disabled_via_env(required_env):
    required_env.setenv("SYFT_ENCLAVE_FRESH_STATE", "false")
    settings = EnclaveSettings(_env_file=None)
    assert settings.fresh_state is False


def test_use_encryption_can_be_disabled_via_env(required_env):
    required_env.setenv("SYFT_ENCLAVE_USE_ENCRYPTION", "false")
    settings = EnclaveSettings(_env_file=None)
    assert settings.use_encryption is False
