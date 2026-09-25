"""attest_peer binds the enclave's attestation to the key bundle the peer holds.

The enclave puts the fingerprint of its identity key into the token's nonce;
a peer must refuse to trust a bundle that arrived over Drive unless it matches.
Runs the real runner and client over a mock Drive; only the launcher socket
and Google's signature check are replaced.
"""

import os
from unittest.mock import patch

import pytest
from attestation_helpers import valid_claims
from syft.sync.peers.peer_store import PeerStore

os.environ["PRE_SYNC"] = "false"

from syft_enclaves import SyftEnclaveClient
from syft_enclaves.attestation import AttestationError, bundle_fingerprint
from syft_enclaves.runner import EnclaveRunner
from syft_enclaves.tee_token import NO_KEY_FINGERPRINT_NONCE, build_eat_nonce


def _publish_token(enclave: SyftEnclaveClient) -> list[str]:
    """Run the runner's attestation step with a fake launcher; return its nonces."""
    with patch("syft_enclaves.runner.fetch_attestation_token") as fetch:
        fetch.return_value = "fake-token"
        EnclaveRunner(client=enclave, fresh_state=False)._publish_attestation()
    return fetch.call_args.kwargs["eat_nonce"]


def _attest(verifier: SyftEnclaveClient, enclave_email: str, eat_nonce: list[str]):
    """Run attest_peer with Google's signature check replaced by fixed claims."""
    with (
        patch("syft_enclaves.attestation.id_token.verify_token") as verify,
        patch("syft_enclaves.attestation.google_requests.Request"),
    ):
        verify.return_value = valid_claims(eat_nonce=eat_nonce)
        return verifier.attest_peer(enclave_email)


def _quad(encryption: bool = True):
    return SyftEnclaveClient.quad_with_mock_drive_service_connection(
        use_in_memory_cache=False, encryption=encryption
    )


def _store(client: SyftEnclaveClient):
    return client._rds.peer_manager.peer_store


def test_attest_peer_verifies_enclave_key_binding():
    enclave, _do1, _do2, ds = _quad()
    eat_nonce = _publish_token(enclave)
    held = _store(ds).get_cached_peer(enclave.email).public_encryption_bundle
    assert eat_nonce[1] == bundle_fingerprint(held)

    result = _attest(ds, enclave.email, eat_nonce)

    key_check = next(c for c in result.checks if c.name == "key_binding")
    assert key_check.passed is True


def _trust_forged_enclave_bundle(ds: SyftEnclaveClient, enclave_email: str) -> dict:
    """Make the DS hold an attacker's bundle for the enclave; return it.

    The bundle names the enclave's identity and is validly self-signed, so it
    passes the checks at ingestion, but its keys are the attacker's. The DS
    already pinned the real key, so ``allow_key_change`` stands in for a DS who
    was talked into trusting the new one.
    """
    attacker = PeerStore(email=enclave_email, use_encryption=True)
    attacker.generate_keys()
    forged = attacker.get_public_bundle()
    _store(ds).set_peer_bundle(enclave_email, forged, allow_key_change=True)
    return forged


def test_attest_peer_rejects_swapped_enclave_bundle():
    """The DS holds an attacker's bundle under the enclave's name: the token does
    not bind that key, so attestation fails on key_binding."""
    enclave, _do1, _do2, ds = _quad()
    eat_nonce = _publish_token(enclave)
    _trust_forged_enclave_bundle(ds, enclave.email)

    with pytest.raises(AttestationError, match="key_binding"):
        _attest(ds, enclave.email, eat_nonce)


def test_attest_peer_rejects_token_without_key_binding():
    enclave, _do1, _do2, ds = _quad()
    eat_nonce = _publish_token(enclave)

    with pytest.raises(AttestationError, match="key_binding"):
        _attest(ds, enclave.email, eat_nonce[:1])


def test_attest_peer_requires_enclave_bundle():
    """Attesting before the enclave's bundle arrived cannot bind anything, so it
    is refused with a hint rather than reported as attested."""
    enclave, _do1, _do2, ds = _quad()
    eat_nonce = _publish_token(enclave)
    _store(ds).get_cached_peer(enclave.email).public_encryption_bundle = None

    with pytest.raises(AttestationError, match="client.sync()"):
        _attest(ds, enclave.email, eat_nonce)


def test_attest_peer_rejects_attacker_key_sent_as_caller_nonce():
    """The attacker swaps in their bundle, then asks the enclave's HTTP server for
    a token with their own fingerprint as the nonce. It lands in the caller slot,
    not the key slot, so the token binds nothing and attestation fails."""
    enclave, _do1, _do2, ds = _quad()
    _publish_token(enclave)
    attacker_bundle = _trust_forged_enclave_bundle(ds, enclave.email)
    eat_nonce = build_eat_nonce(caller_nonce=bundle_fingerprint(attacker_bundle))

    with pytest.raises(AttestationError, match="key_binding"):
        _attest(ds, enclave.email, eat_nonce)


def test_attest_peer_rejects_token_that_binds_no_key():
    enclave, _do1, _do2, ds = _quad()
    _publish_token(enclave)

    with pytest.raises(AttestationError, match="key_binding"):
        _attest(ds, enclave.email, build_eat_nonce())


def test_attest_peer_skips_key_binding_without_encryption():
    enclave, _do1, _do2, ds = _quad(encryption=False)
    eat_nonce = _publish_token(enclave)
    assert eat_nonce[1] == NO_KEY_FINGERPRINT_NONCE

    result = _attest(ds, enclave.email, eat_nonce)

    key_check = next(c for c in result.checks if c.name == "key_binding")
    assert key_check.passed is None
