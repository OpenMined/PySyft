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
    RECEIPT_FILE_NAME,
    ReceiptVerificationError,
    sign_receipt,
    upload_to_rekor,
    verify_receipt,
)
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
    return {"job": {"name": "j"}, "execution": {"runPublicKey": public_key}}


def test_signed_receipt_verifies_and_rejects_tampering():
    jwks, bundle = _keys()
    envelope = sign_receipt(_receipt_for(bundle), jwks)
    assert verify_receipt(envelope, bundle)["job"]["name"] == "j"

    tampered = json.loads(base64.b64decode(envelope["payload"]))
    tampered["job"]["name"] = "other"
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


def _run_job_with_receipts():
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
    code = make_job_code(do1.email, do2.email)
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
    enclave.distribute_results()
    ds.sync()
    return enclave, do1, do2, ds, code, privates


def test_finished_job_ships_a_signed_receipt():
    enclave, do1, do2, ds, code, privates = _run_job_with_receipts()
    job = ds.jobs["test_job"]
    by_name = {p.name: p for p in job.output_paths}
    assert RECEIPT_FILE_NAME in by_name

    bundle = enclave._rds.peer_manager.peer_store.get_public_bundle()
    receipt = verify_receipt(json.loads(by_name[RECEIPT_FILE_NAME].read_text()), bundle)

    assert receipt["dataOwners"] == sorted([do1.email, do2.email])
    assert receipt["job"]["code"][0]["content"] == code
    hashes = {d["name"]: d["files"][0]["sha256"] for d in receipt["datasets"]}
    for name, private in privates.items():
        assert hashes[name] == hashlib.sha256(private.read_bytes()).hexdigest()
    results = {r["path"]: r["content"] for r in receipt["results"]}
    assert results == {"result.json": by_name["result.json"].read_text()}
    execution = receipt["execution"]
    assert execution["platform"] == "local"
    assert execution["startedAt"] and execution["finishedAt"]


def test_execution_on_tinfoil_records_the_config_and_report(tmp_path, monkeypatch):
    from syft_enclaves.receipt import collect

    config = tmp_path / "config.yml"
    config.write_text("cvm-version: 0.14.7\n")
    report = tmp_path / "attestation.json"
    report.write_text(json.dumps({"format": "x/sev-snp-guest/v2", "body": "Zm9v"}))
    monkeypatch.setattr(collect, "TINFOIL_CONFIG_PATH", config)
    monkeypatch.setattr(collect, "TINFOIL_ATTESTATION_PATH", report)
    monkeypatch.setattr(collect.TinfoilProvider, "detect", classmethod(lambda c: True))

    execution = collect.execution_section(None, None, b"\x01", "Org/repo", "v1")

    assert execution["platform"] == "tinfoil-containers"
    assert execution["cvmVersion"] == "0.14.7"
    assert execution["configDigest"] == hashlib.sha256(config.read_bytes()).hexdigest()
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
