import base64
import hashlib
import io
import json
import os
import socketserver
import tempfile
import threading
import urllib.error
from email.message import Message
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

os.environ["PRE_SYNC"] = "false"

from syft_enclaves import SyftEnclaveClient
from syft_enclaves.enclave_job_info import (
    PartyApprovalStatus,
    enclave_approval_file_name,
)
from syft_enclaves.receipt import (
    CLAIMS_FILE_NAME,
    RECEIPT_FILE_NAME,
    ReceiptClaimsError,
    ReceiptVerificationError,
    build_receipt,
    sign_receipt,
    upload_to_rekor,
    verify_receipt,
)
from syft_enclaves.receipt.claims import read_job_claims
from syft_enclaves.receipt.collect import policy_section
from syft_enclaves.receipt.dsse import canonical_json
from syft_enclaves.evidence.tinfoil import fetch_nonce_bound_document
from syft_enclaves.receipt.key_binding import (
    CRYPTO_MATERIAL_V1,
    KEY_SPKI_V1,
    REPORT_DATA_V1,
    bound_key,
    key_binding,
)
from syft_enclaves.receipt.signing_key import ATTESTED_KEY_ID, receipt_signing_key
from syft_enclaves.receipt.writer import ReceiptSettings
from test_enclave_jobs import (
    create_tmp_code_file,
    create_tmp_dataset_files,
    make_job_code,
)


def _signed_receipt(key=None):
    key = key or Ed25519PrivateKey.generate()
    return sign_receipt(_bound_receipt(_attested_document(key.public_key())), key)


def test_signed_receipt_verifies_and_rejects_tampering():
    envelope = _signed_receipt()
    assert envelope["payloadType"] == "application/vnd.in-toto+json"
    assert verify_receipt(envelope)["predicate"]["job"]["name"] == "j"

    tampered = json.loads(base64.b64decode(envelope["payload"]))
    tampered["predicate"]["job"]["name"] = "other"
    envelope["payload"] = base64.b64encode(json.dumps(tampered).encode()).decode()
    with pytest.raises(ReceiptVerificationError):
        verify_receipt(envelope)


def test_receipt_without_a_key_binding_is_rejected():
    key = Ed25519PrivateKey.generate()
    envelope = sign_receipt(build_receipt({}, job={"name": "j"}), key)
    with pytest.raises(ReceiptVerificationError, match="no key binding"):
        verify_receipt(envelope)


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_upload_to_rekor_sends_a_dsse_entry(monkeypatch):
    envelope = _signed_receipt()
    sent = {}

    def urlopen(request, timeout):
        sent["body"] = json.loads(request.data)
        return _Response(json.dumps({"abc": {"logIndex": 7}}).encode())

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    entry = upload_to_rekor(envelope)

    assert entry["uuid"] == "abc" and entry["logIndex"] == 7
    content = sent["body"]["spec"]["proposedContent"]
    assert sent["body"]["kind"] == "dsse"
    assert json.loads(content["envelope"]) == envelope
    assert b"BEGIN PUBLIC KEY" in base64.b64decode(content["verifiers"][0])


def test_upload_to_rekor_returns_the_existing_entry_on_conflict(monkeypatch):
    envelope = _signed_receipt()
    headers = Message()
    headers["Location"] = "/api/v1/log/entries/abc"

    def urlopen(request, timeout):
        if isinstance(request, str):
            return _Response(json.dumps({"abc": {"logIndex": 7}}).encode())
        raise urllib.error.HTTPError("u", 409, "conflict", headers, io.BytesIO())

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    assert upload_to_rekor(envelope)["logIndex"] == 7


CLAIMS = {
    "subject": [{"name": "base + adapter", "digest": {"sha256": "ab" * 32}}],
    "model": {
        "base": {"name": "base"},
        "adapter": {"name": "dataset1"},
        "sampling": {"temperature": 0.8},
    },
    "eval": {"evalSet": {"name": "dataset2"}},
    "results": {"counts": {"submitted": 1, "completed": 1, "failed": 0}},
}

CLAIMS_CODE = f"""
with open("outputs/{CLAIMS_FILE_NAME}", "w") as f:
    f.write(json.dumps({CLAIMS!r}))
"""


def _run_job_with_receipts(before_distribute=None, extra_code=""):
    enclave, do1, do2, ds = SyftEnclaveClient.quad_with_mock_drive_service_connection(
        use_in_memory_cache=False, encryption=True
    )
    enclave.receipts = ReceiptSettings()
    privates = {}
    for owner, name in [(do1, "dataset1"), (do2, "dataset2")]:
        mock, private = create_tmp_dataset_files(name)
        privates[name] = private
        owner.create_dataset(
            name=name,
            mock_path=mock,
            private_path=private,
            summary=name,
            users=[ds.email, enclave.email],
            upload_private=True,
            sync=False,
        )
        owner.share_private_dataset(name, enclave.email)
        owner.sync()
    ds.sync()
    code = make_job_code(do1.email, do2.email) + extra_code
    ds.submit_python_job(
        enclave.email,
        create_tmp_code_file(code),
        "test_job",
        datasets={do1.email: ["dataset1"], do2.email: ["dataset2"]},
    )
    enclave.sync()
    enclave.receive_jobs()
    for owner in (do1, do2):
        owner.sync()
        owner.approve_job(owner.jobs["test_job"])
    enclave.sync()
    enclave.run_jobs()
    if before_distribute is not None:
        before_distribute(enclave, ds)
    enclave.distribute_results()
    ds.sync()
    return enclave, do1, do2, ds, code, privates


def _write_private_pem(path, key):
    path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )


def _write_tinfoil_files(tmp_path):
    config = tmp_path / "config.yml"
    config.write_text("cvm-version: 0.14.12\ncontainers:\n  - image: x/y@sha256:abc\n")
    report = tmp_path / "attestation.json"
    report.write_text(json.dumps({"format": "x/sev-snp-guest/v2", "body": "Zm9v"}))
    return config, report


@pytest.fixture
def mock_tinfoil(tmp_path, monkeypatch):
    """Run as if on Tinfoil: an attested key, its config, report and socket."""
    from syft_enclaves.receipt import collect, signing_key

    key = Ed25519PrivateKey.generate()
    _write_private_pem(tmp_path / "private_key.pem", key)
    config, report = _write_tinfoil_files(tmp_path)
    socket_path = _socket_path()
    monkeypatch.setattr(signing_key, "ATTESTED_KEY_DIR", tmp_path)
    monkeypatch.setattr(collect, "TINFOIL_CONFIG_PATH", config)
    monkeypatch.setattr(collect, "TINFOIL_ATTESTATION_PATH", report)
    monkeypatch.setattr(
        "syft_enclaves.evidence.tinfoil.TINFOIL_ATTESTATION_SOCKET", socket_path
    )
    server = _serve_attestation(socket_path, [], listed_key=key.public_key())
    key_binding.cache_clear()
    yield key
    server.shutdown()
    server.server_close()
    key_binding.cache_clear()


def test_finished_job_ships_a_signed_receipt(mock_tinfoil):
    enclave, do1, do2, ds, code, privates = _run_job_with_receipts(
        extra_code=CLAIMS_CODE
    )
    job = ds.jobs["test_job"]
    by_name = {p.name: p for p in job.output_paths}
    assert RECEIPT_FILE_NAME in by_name

    receipt = verify_receipt(json.loads(by_name[RECEIPT_FILE_NAME].read_text()))

    assert receipt["_type"] == "https://in-toto.io/Statement/v1"
    assert receipt["subject"] == CLAIMS["subject"]
    predicate = receipt["predicate"]
    for key in ("model", "eval", "results"):
        assert predicate[key] == CLAIMS[key]

    assert predicate["job"]["code"][0]["content"] == code
    hashes = {d["name"]: d["files"][0]["sha256"] for d in predicate["datasets"]}
    for name, private in privates.items():
        assert hashes[name] == hashlib.sha256(private.read_bytes()).hexdigest()
    outputs = {r["path"]: r["content"] for r in predicate["outputs"]}
    assert outputs == {"result.json": by_name["result.json"].read_text()}

    execution = predicate["execution"]
    assert execution["platform"] == "tinfoil-containers"
    attested = mock_tinfoil.public_key().public_bytes_raw()
    assert bound_key(execution["attestation"]["keyBinding"]) == attested
    assert execution["startedAt"] and execution["finishedAt"]

    assert predicate["parties"] == [
        {"role": "submitter", "email": ds.email},
        *sorted(
            [
                {"role": "data_owner", "email": do1.email, "datasets": ["dataset1"]},
                {"role": "data_owner", "email": do2.email, "datasets": ["dataset2"]},
            ],
            key=lambda p: p["email"],
        ),
    ]
    consent = predicate["consent"]
    code_digest = hashlib.sha256(canonical_json(predicate["job"]["code"])).hexdigest()
    assert consent["manifestDigest"] == code_digest
    assert {a["party"] for a in consent["approvals"]} == {do1.email, do2.email}
    assert all(a["approvedAt"] for a in consent["approvals"])
    grants = {p["party"]: p["grants"] for p in predicate["policy"]["outputPolicy"]}
    assert grants == {ds.email: ["results", "receipt"], do1.email: [], do2.email: []}


def test_a_job_without_claims_gets_a_receipt_without_them(tmp_path):
    assert read_job_claims(tmp_path) == {}
    receipt = build_receipt({}, job={"name": "j"})
    assert receipt["subject"] == []
    assert set(receipt["predicate"]) == {"job"}


def test_claims_cannot_set_what_the_enclave_writes(tmp_path):
    (tmp_path / CLAIMS_FILE_NAME).write_text(json.dumps({"execution": {}}))
    with pytest.raises(ReceiptClaimsError, match="execution"):
        read_job_claims(tmp_path)

    (tmp_path / CLAIMS_FILE_NAME).write_text("not json")
    with pytest.raises(ReceiptClaimsError, match="not valid JSON"):
        read_job_claims(tmp_path)


def test_policy_grants_owners_the_results_only_when_shared():
    shared = policy_section("ds@x", ["do@x", "ds@x"], share_with_owners=True)
    private = policy_section("ds@x", ["do@x"], share_with_owners=False)

    assert shared["outputPolicy"] == [
        {"party": "ds@x", "grants": ["results", "receipt"]},
        {"party": "do@x", "grants": ["results", "receipt"]},
    ]
    assert private["outputPolicy"][1] == {"party": "do@x", "grants": []}


def test_execution_on_tinfoil_records_the_config_and_report(tmp_path, monkeypatch):
    from syft_enclaves.receipt import collect

    config = tmp_path / "config.yml"
    config.write_text(
        "cvm-version: 0.14.7\ncontainers:\n  - image: docker.io/x/y@sha256:abc\n"
    )
    report = tmp_path / "attestation.json"
    report.write_text(json.dumps({"format": "x/sev-snp-guest/v2", "body": "Zm9v"}))
    monkeypatch.setattr(collect, "TINFOIL_CONFIG_PATH", config)
    monkeypatch.setattr(collect, "TINFOIL_ATTESTATION_PATH", report)
    monkeypatch.setattr(collect, "key_binding", lambda key: {"bound": key.hex()})

    execution = collect.execution_section(None, None, b"\x01", "Org/repo", "v1")

    assert execution["platform"] == "tinfoil-containers"
    assert execution["cvmVersion"] == "0.14.7"
    assert execution["configDigest"] == hashlib.sha256(config.read_bytes()).hexdigest()
    assert execution["runtimeImage"] == {"scheme": "oci/1", "digest": "sha256:abc"}
    assert execution["attestation"]["quote"] == "Zm9v"
    assert execution["attestation"]["referenceValue"]["repo"] == "github.com/Org/repo"
    assert execution["attestation"]["keyBinding"] == {"bound": "01"}


def test_a_failed_receipt_ships_the_error_instead(monkeypatch, mock_tinfoil):
    def fail(*args, **kwargs):
        raise RuntimeError("no key to sign with")

    monkeypatch.setattr("syft_enclaves.client.write_receipt", fail)
    *_, ds, _, _ = _run_job_with_receipts()
    by_name = {p.name: p for p in ds.jobs["test_job"].output_paths}
    assert RECEIPT_FILE_NAME not in by_name
    assert "no key to sign with" in by_name["receipt_error.txt"].read_text()
    assert "result.json" in by_name


def test_receipt_lists_only_approvals_of_the_submission_that_ran(mock_tinfoil):
    def mark_second_approval_stale(enclave, ds):
        review_dir = enclave._local_jobs()["test_job"].job_review_path
        path = review_dir / enclave_approval_file_name(enclave.data_owners[1])
        approval = PartyApprovalStatus.load_json(path)
        approval.submission_hash = "an-earlier-submission"
        approval.save_json(path)

    enclave, *_, ds, _, _ = _run_job_with_receipts(
        before_distribute=mark_second_approval_stale
    )
    by_name = {p.name: p for p in ds.jobs["test_job"].output_paths}
    receipt = verify_receipt(json.loads(by_name[RECEIPT_FILE_NAME].read_text()))

    approvals = receipt["predicate"]["consent"]["approvals"]
    assert [a["party"] for a in approvals] == [enclave.data_owners[0]]


def test_results_do_not_arrive_ahead_of_the_receipt(mock_tinfoil):
    def check(enclave, ds):
        enclave.sync()
        ds.sync()
        assert not ds.jobs["test_job"].output_paths

    _, _, _, ds, _, _ = _run_job_with_receipts(before_distribute=check)
    names = {p.name for p in ds.jobs["test_job"].output_paths}
    assert names == {"result.json", RECEIPT_FILE_NAME}


def _spki_hex(public_key):
    return public_key.public_bytes(
        serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo
    ).hex()


def _crypto_material(public_key, key_id=ATTESTED_KEY_ID):
    item = {"id": key_id, "format": KEY_SPKI_V1, "data": _spki_hex(public_key)}
    return canonical_json({"format": CRYPTO_MATERIAL_V1, "items": [item]})


def _v3_document(nonce_hex, crypto_material=b"{}", device_evidence=b"{}"):
    """A v3 document whose report data is derived from *nonce_hex*, unsigned."""
    digest = hashlib.sha256(REPORT_DATA_V1.encode())
    digest.update(bytes.fromhex(nonce_hex))
    digest.update(hashlib.sha256(crypto_material).digest())
    digest.update(hashlib.sha256(device_evidence).digest())
    return {
        "format": "https://tinfoil.sh/predicate/attestation/v3",
        "challenge": {
            "nonce": nonce_hex,
            "report_data": (digest.digest() + bytes(32)).hex(),
            "report_data_algorithm": REPORT_DATA_V1,
        },
        "crypto_material": base64.b64encode(crypto_material).decode(),
        "device_evidence": base64.b64encode(device_evidence).decode(),
    }


def _attested_document(listed_key, key_id=ATTESTED_KEY_ID):
    return _v3_document("ab" * 32, _crypto_material(listed_key, key_id))


def _bound_receipt(document):
    execution = {"attestation": {"keyBinding": document}}
    return build_receipt({}, job={"name": "j"}, execution=execution)


def test_receipt_signed_by_another_key_is_rejected():
    key = Ed25519PrivateKey.generate()
    document = _attested_document(key.public_key())
    forger = Ed25519PrivateKey.generate()
    with pytest.raises(ReceiptVerificationError, match="no signature"):
        verify_receipt(sign_receipt(_bound_receipt(document), forger))


def test_key_binding_that_does_not_list_the_receipt_key_is_rejected():
    key = Ed25519PrivateKey.generate()

    other_id = _attested_document(key.public_key(), key_id="other-key")
    with pytest.raises(ReceiptVerificationError, match="no attested key"):
        verify_receipt(sign_receipt(_bound_receipt(other_id), key))

    tampered = _attested_document(key.public_key())
    tampered["challenge"]["report_data"] = "00" * 64
    with pytest.raises(ReceiptVerificationError, match="report data"):
        verify_receipt(sign_receipt(_bound_receipt(tampered), key))


def test_receipts_are_signed_with_the_attested_key(tmp_path, monkeypatch):
    from syft_enclaves.receipt import signing_key

    monkeypatch.setattr(signing_key, "ATTESTED_KEY_DIR", tmp_path)
    with pytest.raises(RuntimeError, match="No attested key"):
        receipt_signing_key()

    key = Ed25519PrivateKey.generate()
    _write_private_pem(tmp_path / "private_key.pem", key)
    signing = receipt_signing_key()
    assert (
        signing.public_key().public_bytes_raw() == key.public_key().public_bytes_raw()
    )


def _serve_attestation(socket_path, requests, listed_key=None):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            nonce = self.path.split("nonce=", 1)[1]
            material = _crypto_material(listed_key) if listed_key else b"{}"
            body = json.dumps(_v3_document(nonce, material)).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    class Server(socketserver.UnixStreamServer):
        def get_request(self):
            request, _ = super().get_request()
            return request, ("local", 0)

    server = Server(str(socket_path), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _socket_path():
    # AF_UNIX paths are capped near 104 bytes, too short for macOS tmp_path.
    return Path(tempfile.mkdtemp(dir="/tmp")) / "a.sock"


def test_bound_document_is_fetched_from_the_local_socket_for_our_nonce():
    socket_path = _socket_path()
    requests = []
    server = _serve_attestation(socket_path, requests)
    try:
        nonce = "ab" * 32
        document = fetch_nonce_bound_document(nonce, socket_path)
    finally:
        server.shutdown()
        server.server_close()
    assert requests == [f"/.well-known/tinfoil-attestation?nonce={nonce}"]
    assert document["challenge"]["nonce"] == nonce


def test_key_binding_names_the_attested_key(monkeypatch):
    key = Ed25519PrivateKey.generate().public_key()
    socket_path = _socket_path()
    monkeypatch.setattr(
        "syft_enclaves.evidence.tinfoil.TINFOIL_ATTESTATION_SOCKET", socket_path
    )
    server = _serve_attestation(socket_path, [], listed_key=key)
    key_binding.cache_clear()
    try:
        document = key_binding(key.public_bytes_raw())
        with pytest.raises(RuntimeError, match="does not list"):
            key_binding(b"\x00" * 32)
    finally:
        server.shutdown()
        server.server_close()
    assert bound_key(document) == key.public_bytes_raw()


def test_receipt_is_not_signed_when_the_socket_is_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "syft_enclaves.evidence.tinfoil.TINFOIL_ATTESTATION_SOCKET",
        tmp_path / "missing.sock",
    )
    key_binding.cache_clear()
    with pytest.raises(RuntimeError, match="key binding report"):
        key_binding(b"run key")
