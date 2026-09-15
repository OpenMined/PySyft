"""Fetching a Tinfoil enclave's attestation over a connection pinned to it.

Why not plain HTTPS: the enclave serves a self-signed certificate, because its
TLS key is generated inside the enclave and no CA is involved. Validating that
certificate the usual way is therefore impossible, and trusting it blindly
would let anyone in the path serve a replayed report along with their own keys.

What makes this sound is the report itself. A SEV-SNP report's 64 bytes of user
data are the sha256 of the shim's TLS public key followed by its HPKE public
key, so the report *commits to the key terminating the connection*. Verify the
report, then check that the certificate we were served carries that key, and
the channel provably ends inside the attested enclave — no CA required. Whatever
else came down it, notably the enclave's syft key bundle, is then bound to the
hardware report too.

Two useful consequences: the transport does not have to be trusted, and replay
stops working — a captured report commits to a TLS key whose private half lives
in an enclave the attacker does not control.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import ssl
from dataclasses import dataclass
from typing import Any, Optional

from syft_enclaves.nonce_challenge import new_nonce

ATTESTATION_PATH = "/attestation"
WELL_KNOWN_PATH = "/.well-known/tinfoil-attestation"
DEFAULT_TIMEOUT_SECONDS = 30


@dataclass(frozen=True)
class AttestedPayload:
    """What the enclave served, and the key that terminated the connection.

    Untrusted until :mod:`syft_enclaves.attestation_tinfoil` has checked
    ``tls_public_key_fp`` against the verified report and the signature over
    ``nonce`` against ``key_bundle``.
    """

    document: dict[str, Any]
    key_bundle: Optional[dict[str, Any]]
    #: sha256 of the served certificate's DER SubjectPublicKeyInfo.
    tls_public_key_fp: str
    host: str
    #: The nonce we sent, and the enclave's signature over it.
    nonce: str = ""
    nonce_signature: Optional[str] = None


class AttestationFetchError(RuntimeError):
    """The enclave could not be reached, or served something unusable."""


def fetch_attested_payload(
    host: str,
    nonce: Optional[str] = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> AttestedPayload:
    """GET the enclave's attestation and key bundle, recording its TLS key.

    Certificate validation is deliberately disabled: the certificate is
    self-signed, and it is the *report* that decides whether to trust the key.
    Nothing here is trusted — the caller must verify both the fingerprint and
    the signature over *nonce*.

    A nonce is generated when none is given, so a caller cannot accidentally
    skip the freshness proof.
    """
    nonce = nonce or new_nonce()
    connection = http.client.HTTPSConnection(
        host, timeout=timeout, context=_unverified_context()
    )
    try:
        payload = _attestation_or_empty(connection, nonce)
        document, key_bundle = _split_payload(payload, connection, timeout)
        fingerprint = _peer_public_key_fp(connection)
    except (OSError, ssl.SSLError, ValueError) as e:
        raise AttestationFetchError(f"Could not fetch attestation from {host}: {e}") from e
    finally:
        connection.close()

    echoed = payload.get("nonce")
    if echoed is not None and echoed != nonce:
        # Never verify against a nonce the responder chose.
        raise AttestationFetchError(
            f"{host} echoed a different nonce than the one sent"
        )
    return AttestedPayload(
        document=document,
        key_bundle=key_bundle,
        tls_public_key_fp=fingerprint,
        host=host,
        nonce=nonce,
        nonce_signature=payload.get("nonce_signature"),
    )


def _attestation_or_empty(
    connection: http.client.HTTPSConnection, nonce: str
) -> dict[str, Any]:
    """The /attestation response, or {} if the enclave will not serve it.

    An enclave running an older image rejects the nonce outright. Rather than
    surface its 500, fall through to the shim's own endpoint for the report —
    the nonce check then fails with "no signature", which says plainly what is
    wrong instead of hiding it behind a transport error.
    """
    try:
        return _get_json(connection, f"{ATTESTATION_PATH}?nonce={nonce}")
    except ValueError:
        return {}


def _split_payload(
    payload: dict[str, Any], connection: http.client.HTTPSConnection, timeout: float
) -> tuple[dict[str, Any], Optional[dict[str, Any]]]:
    """Pull the raw document and key bundle out of an /attestation response.

    Falls back to the shim's well-known path when the enclave does not run
    syft's own endpoint — that still yields a verifiable report, just no key
    bundle.
    """
    evidence = payload.get("evidence") or {}
    document = {k: evidence[k] for k in ("format", "body") if k in evidence}
    if not document:
        document = _get_json(connection, WELL_KNOWN_PATH)
    if "format" not in document or "body" not in document:
        raise ValueError("response carried no attestation document")
    return document, payload.get("key_bundle")


def _get_json(connection: http.client.HTTPSConnection, path: str) -> dict[str, Any]:
    connection.request("GET", path)
    response = connection.getresponse()
    body = response.read()
    if response.status != 200:
        raise ValueError(f"GET {path} returned {response.status}")
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as e:
        raise ValueError(f"GET {path} did not return JSON: {e}") from e
    if not isinstance(parsed, dict):
        raise ValueError(f"GET {path} did not return an object")
    return parsed


def _unverified_context() -> ssl.SSLContext:
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    return context


def _peer_public_key_fp(connection: http.client.HTTPSConnection) -> str:
    """sha256 of the served certificate's public key, as the report encodes it."""
    socket = connection.sock
    if socket is None:
        raise ValueError("connection closed before the certificate could be read")
    der = socket.getpeercert(binary_form=True)
    if not der:
        raise ValueError("peer presented no certificate")
    return public_key_fp_from_cert(der)


def public_key_fp_from_cert(der_certificate: bytes) -> str:
    """sha256 over the certificate's DER SubjectPublicKeyInfo.

    Computed here rather than imported from ``tinfoil.attestation.attestation``:
    the equivalent helper there is not in that package's ``__all__``, so it is
    not a stable API to depend on.
    """
    from cryptography import x509
    from cryptography.hazmat.primitives import serialization

    certificate = x509.load_der_x509_certificate(der_certificate)
    spki = certificate.public_key().public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return hashlib.sha256(spki).hexdigest()
