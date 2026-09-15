"""Tests for the pinned fetch itself."""
import json
from unittest.mock import MagicMock

import pytest

from syft_enclaves.attestation_https import (
    AttestationFetchError,
    fetch_attested_payload,
    public_key_fp_from_cert,
)

DOC = {"format": "https://tinfoil.sh/predicate/sev-snp-guest/v2", "body": "H4sIA"}


class _FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status = status

    def read(self):
        return self._payload if isinstance(self._payload, bytes) else json.dumps(self._payload).encode()


def _connection(responses, der=b"\x30\x00"):
    conn = MagicMock()
    conn.getresponse.side_effect = responses
    conn.sock.getpeercert.return_value = der
    return conn


@pytest.fixture
def patched(monkeypatch):
    def _install(responses, der=b"\x30\x00"):
        conn = _connection(responses, der)
        monkeypatch.setattr(
            "syft_enclaves.attestation_https.http.client.HTTPSConnection",
            lambda *a, **k: conn,
        )
        monkeypatch.setattr(
            "syft_enclaves.attestation_https.public_key_fp_from_cert",
            lambda der_bytes: "ab" * 32,
        )
        return conn

    return _install


def test_returns_document_bundle_and_tls_fingerprint(patched):
    bundle = {"identity": "enclave@openmined.org"}
    patched([_FakeResponse({"evidence": {**DOC, "kind": "tinfoil"}, "key_bundle": bundle})])
    payload = fetch_attested_payload("enclave.example")
    assert payload.document == DOC
    assert payload.key_bundle == bundle
    assert payload.tls_public_key_fp == "ab" * 32


def test_falls_back_to_the_shim_well_known_path(patched):
    # An enclave not running syft's endpoint still yields a verifiable report.
    patched([_FakeResponse({"status": "weird"}), _FakeResponse(DOC)])
    payload = fetch_attested_payload("enclave.example")
    assert payload.document == DOC
    assert payload.key_bundle is None


def test_a_missing_certificate_is_an_error(patched):
    patched([_FakeResponse({"evidence": DOC})], der=None)
    with pytest.raises(AttestationFetchError):
        fetch_attested_payload("enclave.example")


def test_an_error_from_both_endpoints_is_an_error(patched):
    patched([_FakeResponse({}, status=503), _FakeResponse({}, status=503)])
    with pytest.raises(AttestationFetchError):
        fetch_attested_payload("enclave.example")


def test_non_json_from_both_endpoints_is_an_error(patched):
    patched([_FakeResponse(b"<html>nope</html>"), _FakeResponse(b"<html>nope</html>")])
    with pytest.raises(AttestationFetchError):
        fetch_attested_payload("enclave.example")


def test_an_older_enclave_that_rejects_the_nonce_still_yields_a_report(patched):
    """It 500s on /attestation?nonce=, so fall through to the shim endpoint.

    The nonce check then fails with "no signature", which names the problem
    instead of hiding it behind a transport error.
    """
    patched([_FakeResponse({}, status=500), _FakeResponse(DOC)])
    payload = fetch_attested_payload("enclave.example")
    assert payload.document == DOC
    assert payload.nonce_signature is None


def test_a_nonce_is_always_sent(patched):
    conn = patched([_FakeResponse({"evidence": DOC})])
    payload = fetch_attested_payload("enclave.example")
    assert payload.nonce
    assert f"nonce={payload.nonce}" in conn.request.call_args_list[0].args[1]


def test_an_echoed_nonce_that_differs_is_refused(patched):
    # Verifying against a nonce the responder chose would prove nothing.
    patched([_FakeResponse({"evidence": DOC, "nonce": "not-the-one-we-sent"})])
    with pytest.raises(AttestationFetchError, match="different nonce"):
        fetch_attested_payload("enclave.example")


def test_fingerprint_is_the_spki_sha256_of_a_real_certificate():
    # Not mocked: the fingerprint must match what the report encodes.
    import datetime, hashlib
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "enclave")])
    now = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(1)
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(days=1))
        .sign(key, hashes.SHA256())
    )
    expected = hashlib.sha256(
        key.public_key().public_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    ).hexdigest()
    assert public_key_fp_from_cert(cert.public_bytes(serialization.Encoding.DER)) == expected
