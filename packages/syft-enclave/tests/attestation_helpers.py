"""Shared fixtures for attestation tests: fake claims and the values they carry."""

from syft.version import SYFT_VERSION

from syft_enclaves.attestation import AppraisalPolicy

FAKE_IMAGE_DIGEST = "sha256:abc123"
EXPECTED_VERSION_NONCE = f"syft-{SYFT_VERSION}"
FAKE_KEY_FINGERPRINT = "ab" * 32

# The image digest is not shipped as a constant — it's supplied per-call via an
# AppraisalPolicy. A policy pinning the fake token's digest and key is used by
# the tests that need every check to pass.
DEFAULT_TEST_POLICY = AppraisalPolicy(
    expected_image_digest=FAKE_IMAGE_DIGEST,
    expected_key_fingerprint=FAKE_KEY_FINGERPRINT,
)


def valid_claims(**overrides):
    """Build a valid claims dict, optionally overriding specific fields."""
    claims = {
        "secboot": True,
        "dbgstat": "disabled-since-boot",
        "eat_nonce": [EXPECTED_VERSION_NONCE, FAKE_KEY_FINGERPRINT],
        "submods": {
            "container": {
                "image_digest": FAKE_IMAGE_DIGEST,
                "image_reference": "docker.io/openmined/syft-enclave:latest",
            }
        },
    }
    claims.update(overrides)
    return claims
