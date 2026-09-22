"""The nonces the enclave asks the Confidential Space launcher to sign."""

import json
from unittest.mock import MagicMock, patch

import syft_crypto_python as syc
from attestation_helpers import EXPECTED_VERSION_NONCE, FAKE_KEY_FINGERPRINT

from syft_enclaves.tee_token import (
    KEY_FINGERPRINT_NONCE_SLOT,
    VERSION_NONCE_SLOT,
    build_eat_nonce,
    fetch_attestation_token,
    validate_nonce,
)


def test_build_eat_nonce_version_only():
    assert build_eat_nonce() == [EXPECTED_VERSION_NONCE]


def test_build_eat_nonce_puts_key_fingerprint_in_its_slot():
    nonces = build_eat_nonce(key_fingerprint=FAKE_KEY_FINGERPRINT)
    assert nonces[VERSION_NONCE_SLOT] == EXPECTED_VERSION_NONCE
    assert nonces[KEY_FINGERPRINT_NONCE_SLOT] == FAKE_KEY_FINGERPRINT


def test_build_eat_nonce_caller_nonce_follows_key_fingerprint():
    nonces = build_eat_nonce(caller_nonce="fresh-1234", key_fingerprint="ab" * 32)
    assert nonces == [EXPECTED_VERSION_NONCE, "ab" * 32, "fresh-1234"]


def test_build_eat_nonce_caller_nonce_without_key_takes_next_slot():
    """The diagnostic HTTP server passes no key; its caller nonce lands in slot 1."""
    assert build_eat_nonce(caller_nonce="fresh-1234") == [
        EXPECTED_VERSION_NONCE,
        "fresh-1234",
    ]


def test_real_key_fingerprint_is_a_valid_nonce():
    fingerprint = (
        syc.SyftRecoveryKey.generate().derive_keys().to_public_bundle()
    ).identity_fingerprint()
    assert validate_nonce(fingerprint) is None


def test_fetch_attestation_token_posts_nonces():
    response = MagicMock(status=200)
    response.read.return_value = b"signed.jwt.token\n"
    with patch("syft_enclaves.tee_token._UnixSocketConnection") as connection_cls:
        connection_cls.return_value.getresponse.return_value = response
        token = fetch_attestation_token(eat_nonce=[EXPECTED_VERSION_NONCE, "ab" * 32])

    assert token == "signed.jwt.token"
    _, kwargs = connection_cls.return_value.request.call_args
    assert json.loads(kwargs["body"])["nonces"] == [EXPECTED_VERSION_NONCE, "ab" * 32]
