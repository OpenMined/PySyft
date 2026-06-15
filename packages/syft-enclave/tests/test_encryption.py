"""Encryption tests for the enclave flow."""

import os
import random
import tempfile
from pathlib import Path

os.environ["PRE_SYNC"] = "false"

from syft_enclaves import SyftEnclaveClient


def _simple_job_code() -> str:
    return (
        "import json, os\n"
        'os.makedirs("outputs", exist_ok=True)\n'
        'open("outputs/result.json", "w").write(json.dumps({"ok": 1}))\n'
    )


def create_tmp_code_file(code: str) -> str:
    tmp_dir = Path(tempfile.mkdtemp()) / f"syft-enc-code-{random.randint(1, 1000000)}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    code_path = tmp_dir / "main.py"
    code_path.write_text(code)
    return str(code_path)


def test_encrypted_quad_keys_and_bundles():
    """encryption=True keys every client and exchanges public-key bundles."""
    enclave, do1, do2, ds = SyftEnclaveClient.quad_with_mock_drive_service_connection(
        use_in_memory_cache=False,
        encryption=True,
    )

    for client in (enclave, do1, do2, ds):
        store = client._manager._peer_store
        assert store is not None
        assert store.use_encryption is True
        assert store.has_my_keys() is True

    # The enclave peers with all three; it should hold each of their bundles
    # after wiring, so it can encrypt to and verify from them.
    enclave_store = enclave._manager._peer_store
    for peer_email in (do1.email, do2.email, ds.email):
        assert enclave_store.has_peer_bundle(peer_email), (
            f"enclave missing encryption bundle for {peer_email}"
        )

    # And the DS holds the enclave's bundle (the link we care most about).
    assert ds._manager._peer_store.has_peer_bundle(enclave.email)


def test_encrypted_message_to_enclave_roundtrips():
    """A job submitted to the enclave is encrypted on the wire and decrypted by
    the enclave — the user's core goal ("messages sent to the enclave are
    encrypted"). Uses a dataset-free job to isolate the message path."""
    enclave, _do1, _do2, ds = SyftEnclaveClient.quad_with_mock_drive_service_connection(
        use_in_memory_cache=False,
        encryption=True,
    )

    code_path = create_tmp_code_file(_simple_job_code())
    ds.submit_python_job(enclave.email, code_path, "simple_job")

    # Enclave pulls the ProposedFileChanges message from its inbox and decrypts
    # it; if the message weren't a valid SYC envelope, this would raise.
    enclave.sync()

    assert "simple_job" in [j.name for j in enclave.jobs]
