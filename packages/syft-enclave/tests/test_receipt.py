import base64
import hashlib
import io
import json
import os
import urllib.error
from email.message import Message

import pytest

os.environ["PRE_SYNC"] = "false"

from syft_enclaves import SyftEnclaveClient
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
from syft_enclaves.receipt.writer import ReceiptSettings
from test_enclave_jobs import (
    create_tmp_code_file,
    create_tmp_dataset_files,
    make_job_code,
)


def _keys():
    import syft_crypto_python as syc

    keys = syc.SyftRecoveryKey.generate().derive_keys()
    bundle = keys.to_public_bundle().to_did_document("did:syft:enclave@openmined.org")
    return keys.to_jwks(), bundle


def _receipt_for(bundle):
    from syft_enclaves.attestation.nonce import identity_key_bytes

    public_key = identity_key_bytes(bundle).hex()
    return build_receipt({}, job={"name": "j"}, execution={"runPublicKey": public_key})


def test_signed_receipt_verifies_and_rejects_tampering():
    jwks, bundle = _keys()
    envelope = sign_receipt(_receipt_for(bundle), jwks)
    assert envelope["payloadType"] == "application/vnd.in-toto+json"
    assert verify_receipt(envelope, bundle)["predicate"]["job"]["name"] == "j"

    tampered = json.loads(base64.b64decode(envelope["payload"]))
    tampered["predicate"]["job"]["name"] = "other"
    envelope["payload"] = base64.b64encode(json.dumps(tampered).encode()).decode()
    with pytest.raises(ReceiptVerificationError):
        verify_receipt(envelope, bundle)


def test_receipt_from_another_key_is_rejected():
    jwks, bundle = _keys()
    _, other_bundle = _keys()
    envelope = sign_receipt(_receipt_for(bundle), jwks)
    with pytest.raises(ReceiptVerificationError):
        verify_receipt(envelope, other_bundle)


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_upload_to_rekor_sends_a_dsse_entry(monkeypatch):
    jwks, bundle = _keys()
    envelope = sign_receipt(_receipt_for(bundle), jwks)
    sent = {}

    def urlopen(request, timeout):
        sent["body"] = json.loads(request.data)
        return _Response(json.dumps({"abc": {"logIndex": 7}}).encode())

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    entry = upload_to_rekor(envelope, bundle)

    assert entry["uuid"] == "abc" and entry["logIndex"] == 7
    content = sent["body"]["spec"]["proposedContent"]
    assert sent["body"]["kind"] == "dsse"
    assert json.loads(content["envelope"]) == envelope
    assert b"BEGIN PUBLIC KEY" in base64.b64decode(content["verifiers"][0])


def test_upload_to_rekor_returns_the_existing_entry_on_conflict(monkeypatch):
    jwks, bundle = _keys()
    envelope = sign_receipt(_receipt_for(bundle), jwks)
    headers = Message()
    headers["Location"] = "/api/v1/log/entries/abc"

    def urlopen(request, timeout):
        if isinstance(request, str):
            return _Response(json.dumps({"abc": {"logIndex": 7}}).encode())
        raise urllib.error.HTTPError("u", 409, "conflict", headers, io.BytesIO())

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    assert upload_to_rekor(envelope, bundle)["logIndex"] == 7


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


def test_finished_job_ships_a_signed_receipt():
    enclave, do1, do2, ds, code, privates = _run_job_with_receipts(
        extra_code=CLAIMS_CODE
    )
    job = ds.jobs["test_job"]
    by_name = {p.name: p for p in job.output_paths}
    assert RECEIPT_FILE_NAME in by_name

    bundle = enclave._rds.peer_manager.peer_store.get_public_bundle()
    receipt = verify_receipt(json.loads(by_name[RECEIPT_FILE_NAME].read_text()), bundle)

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
    assert execution["platform"] == "local"
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
    monkeypatch.setattr(collect.TinfoilProvider, "detect", classmethod(lambda c: True))

    execution = collect.execution_section(None, None, b"\x01", "Org/repo", "v1")

    assert execution["platform"] == "tinfoil-containers"
    assert execution["cvmVersion"] == "0.14.7"
    assert execution["configDigest"] == hashlib.sha256(config.read_bytes()).hexdigest()
    assert execution["runtimeImage"] == {"scheme": "oci/1", "digest": "sha256:abc"}
    assert execution["attestation"]["quote"] == "Zm9v"
    assert execution["attestation"]["referenceValue"]["repo"] == "github.com/Org/repo"


def test_a_failed_receipt_ships_the_error_instead(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("no key to sign with")

    monkeypatch.setattr("syft_enclaves.client.write_receipt", fail)
    *_, ds, _, _ = _run_job_with_receipts()
    by_name = {p.name: p for p in ds.jobs["test_job"].output_paths}
    assert RECEIPT_FILE_NAME not in by_name
    assert "no key to sign with" in by_name["receipt_error.txt"].read_text()
    assert "result.json" in by_name


def test_results_do_not_arrive_ahead_of_the_receipt():
    def check(enclave, ds):
        enclave.sync()
        ds.sync()
        assert not ds.jobs["test_job"].output_paths

    _, _, _, ds, _, _ = _run_job_with_receipts(before_distribute=check)
    names = {p.name for p in ds.jobs["test_job"].output_paths}
    assert names == {"result.json", RECEIPT_FILE_NAME}
