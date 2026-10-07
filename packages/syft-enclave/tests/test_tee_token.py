"""The nonces the enclave asks the Confidential Space launcher to sign."""

import json
from unittest.mock import MagicMock, patch

from syft.version import SYFT_VERSION

from syft_enclaves.evidence.tee_token import (
    CLAIMS_DIGEST_NONCE_SLOT,
    VERSION_NONCE_SLOT,
    build_eat_nonce,
    fetch_attestation_token,
)

EXPECTED_VERSION_NONCE = f"syft-{SYFT_VERSION}"
FAKE_CLAIMS_DIGEST = "ab" * 32


def test_build_eat_nonce_without_claims_holds_only_the_version():
    assert build_eat_nonce() == [EXPECTED_VERSION_NONCE]


def test_build_eat_nonce_puts_the_claims_digest_in_its_slot():
    nonces = build_eat_nonce(FAKE_CLAIMS_DIGEST)
    assert nonces[VERSION_NONCE_SLOT] == EXPECTED_VERSION_NONCE
    assert nonces[CLAIMS_DIGEST_NONCE_SLOT] == FAKE_CLAIMS_DIGEST


def test_fetch_attestation_token_posts_nonces():
    response = MagicMock(status=200)
    response.read.return_value = b"signed.jwt.token\n"
    with patch(
        "syft_enclaves.evidence.tee_token.UnixSocketConnection"
    ) as connection_cls:
        connection_cls.return_value.getresponse.return_value = response
        nonces = [EXPECTED_VERSION_NONCE, FAKE_CLAIMS_DIGEST]
        token = fetch_attestation_token(eat_nonce=nonces)

    assert token == "signed.jwt.token"
    _, kwargs = connection_cls.return_value.request.call_args
    assert json.loads(kwargs["body"])["nonces"] == nonces
