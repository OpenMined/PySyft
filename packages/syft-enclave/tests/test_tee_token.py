"""The nonces the enclave asks the Confidential Space launcher to sign."""

import json
from unittest.mock import MagicMock, patch

import syft_crypto_python as syc
from attestation_helpers import EXPECTED_VERSION_NONCE, FAKE_KEY_FINGERPRINT

from syft_enclaves.tee_token import (
    CALLER_NONCE_SLOT,
    KEY_FINGERPRINT_NONCE_SLOT,
    NO_KEY_FINGERPRINT_NONCE,
    VERSION_NONCE_SLOT,
    build_eat_nonce,
    fetch_attestation_token,
    validate_nonce,
)


def test_build_eat_nonce_without_key_holds_placeholder():
    assert build_eat_nonce() == [EXPECTED_VERSION_NONCE, NO_KEY_FINGERPRINT_NONCE]


def test_build_eat_nonce_puts_key_fingerprint_in_its_slot():
    nonces = build_eat_nonce(key_fingerprint=FAKE_KEY_FINGERPRINT)
    assert nonces[VERSION_NONCE_SLOT] == EXPECTED_VERSION_NONCE
    assert nonces[KEY_FINGERPRINT_NONCE_SLOT] == FAKE_KEY_FINGERPRINT


def test_build_eat_nonce_caller_nonce_follows_key_fingerprint():
    nonces = build_eat_nonce(
        caller_nonce="fresh-1234", key_fingerprint=FAKE_KEY_FINGERPRINT
    )
    assert nonces == [EXPECTED_VERSION_NONCE, FAKE_KEY_FINGERPRINT, "fresh-1234"]


def test_caller_nonce_cannot_take_key_fingerprint_slot():
    """The HTTP server passes any caller's nonce and no key.

    A caller who sends a key fingerprint as the nonce must not see it in the
    key slot, or they would hold a TEE-signed token binding their own key.
    """
    nonces = build_eat_nonce(caller_nonce=FAKE_KEY_FINGERPRINT)
    assert nonces[KEY_FINGERPRINT_NONCE_SLOT] == NO_KEY_FINGERPRINT_NONCE
    assert nonces[CALLER_NONCE_SLOT] == FAKE_KEY_FINGERPRINT


def test_no_key_placeholder_is_valid_nonce():
    assert validate_nonce(NO_KEY_FINGERPRINT_NONCE) is None
    assert len(NO_KEY_FINGERPRINT_NONCE.encode()) >= 8


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
        nonces = [EXPECTED_VERSION_NONCE, FAKE_KEY_FINGERPRINT]
        token = fetch_attestation_token(eat_nonce=nonces)

    assert token == "signed.jwt.token"
    _, kwargs = connection_cls.return_value.request.call_args
    assert json.loads(kwargs["body"])["nonces"] == nonces
