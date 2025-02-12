import base64
import hashlib
from pathlib import Path

import py_fast_rsync
import pytest
import yaml
from fastapi.testclient import TestClient
from py_fast_rsync import signature

from syftbox.client.exceptions import SyftServerError
from syftbox.client.server_client import SyncClient
from syftbox.lib.constants import PERM_FILE
from syftbox.server.models.sync_models import ApplyDiffResponse, DiffResponse, FileMetadata
from tests.unit.server.conftest import TEST_DATASITE_NAME, TEST_FILE


def test_get_diff_2(client: TestClient):
    local_data = b"This is my local data"
    sig = signature.calculate(local_data)
    sig_b85 = base64.b85encode(sig).decode("utf-8")
    response = client.post(
        "/sync/get_diff",
        json={
            "path": f"{TEST_DATASITE_NAME}/{TEST_FILE}",
            "signature": sig_b85,
        },
    )

    response.raise_for_status()
    diff_response = DiffResponse.model_validate(response.json())
    remote_diff = diff_response.diff_bytes
    probably_remote_data = py_fast_rsync.apply(local_data, remote_diff)

    server_settings = client.app_state["server_settings"]
    file_server_contents = server_settings.read(f"{TEST_DATASITE_NAME}/{TEST_FILE}")
    assert file_server_contents == probably_remote_data


def file_digest(file_path, algorithm="sha256"):
    # because this doesnt work in python <=3.10, we implement it manually
    hash_func = hashlib.new(algorithm)

    with open(file_path, "rb") as file:
        # Read the file in chunks to handle large files efficiently
        for chunk in iter(lambda: file.read(4096), b""):
            hash_func.update(chunk)

    return hash_func.hexdigest()


def test_syft_client_push_flow(client: TestClient):
    response = client.post(
        "/sync/get_metadata",
        json={"path": f"{TEST_DATASITE_NAME}/{TEST_FILE}"},
    )

    response.raise_for_status()
    server_signature_b85 = response.json()["signature"]
    server_signature = base64.b85decode(server_signature_b85)
    assert server_signature

    local_data = b"This is my local data"
    delta = py_fast_rsync.diff(server_signature, local_data)
    delta_b85 = base64.b85encode(delta).decode("utf-8")
    expected_hash = hashlib.sha256(local_data).hexdigest()

    response = client.post(
        "/sync/apply_diff",
        json={
            "path": f"{TEST_DATASITE_NAME}/{TEST_FILE}",
            "diff": delta_b85,
            "expected_hash": expected_hash,
        },
    )

    response.raise_for_status()

    result = response.json()
    snapshot_folder = client.app_state["server_settings"].snapshot_folder
    sha256local = file_digest(f"{snapshot_folder}/{TEST_DATASITE_NAME}/{TEST_FILE}", "sha256")
    assert result["current_hash"] == expected_hash == sha256local


def test_get_remote_state(sync_client: SyncClient):
    metadata = sync_client.get_remote_state(Path(TEST_DATASITE_NAME))

    assert len(metadata) == 3


def test_get_metadata(sync_client: SyncClient):
    metadata = sync_client.get_metadata(Path(TEST_DATASITE_NAME) / TEST_FILE)
    assert metadata.path == Path(TEST_DATASITE_NAME) / TEST_FILE

    # check serde works
    assert isinstance(metadata.hash_bytes, bytes)
    assert isinstance(metadata.signature_bytes, bytes)


def test_apply_diff(sync_client: SyncClient):
    local_data = b"This is my local data"

    remote_metadata = sync_client.get_metadata(Path(TEST_DATASITE_NAME) / TEST_FILE)

    diff = py_fast_rsync.diff(remote_metadata.signature_bytes, local_data)
    expected_hash = hashlib.sha256(local_data).hexdigest()

    # Apply local_data to server
    response = sync_client.apply_diff(Path(TEST_DATASITE_NAME) / TEST_FILE, diff, expected_hash)
    assert response.current_hash == expected_hash

    # check file was written correctly
    snapshot_folder = sync_client.conn.app_state["server_settings"].snapshot_folder
    snapshot_file_path = snapshot_folder / Path(TEST_DATASITE_NAME) / TEST_FILE
    remote_data = snapshot_file_path.read_bytes()
    assert local_data == remote_data

    # another diff with incorrect hash
    remote_metadata = sync_client.get_metadata(Path(TEST_DATASITE_NAME) / TEST_FILE)
    diff = py_fast_rsync.diff(remote_metadata.signature_bytes, local_data)
    wrong_hash = "wrong_hash"

    with pytest.raises(SyftServerError):
        sync_client.apply_diff(Path(TEST_DATASITE_NAME) / TEST_FILE, diff, wrong_hash)


def test_get_diff(sync_client: SyncClient):
    local_data = b"This is my local data"
    sig = signature.calculate(local_data)

    file_path = Path(TEST_DATASITE_NAME) / TEST_FILE
    response = sync_client.get_diff(file_path, sig)
    assert response.path == file_path

    # apply and check hash
    new_data = py_fast_rsync.apply(local_data, base64.b85decode(response.diff))
    new_hash = hashlib.sha256(new_data).hexdigest()

    assert new_hash == response.hash

    # diff nonexistent file
    file_path = Path(TEST_DATASITE_NAME) / "nonexistent_file.txt"
    with pytest.raises(SyftServerError):
        sync_client.get_diff(file_path, sig)


def test_delete_file(sync_client: SyncClient):
    sync_client.delete(Path(TEST_DATASITE_NAME) / TEST_FILE)

    snapshot_folder = sync_client.conn.app_state["server_settings"].snapshot_folder
    path = Path(f"{snapshot_folder}/{TEST_DATASITE_NAME}/{TEST_FILE}")
    assert not path.exists()

    with pytest.raises(SyftServerError):
        sync_client.get_metadata(Path(TEST_DATASITE_NAME) / TEST_FILE)


def test_create_file(sync_client: SyncClient):
    snapshot_folder = sync_client.conn.app_state["server_settings"].snapshot_folder
    new_fname = "new.txt"
    contents = b"Some content"
    path = Path(f"{snapshot_folder}/{TEST_DATASITE_NAME}/{new_fname}")
    assert not path.exists()

    with open(path, "wb") as f:
        f.write(contents)

    with open(path, "rb") as f:
        sync_client.create(relative_path=Path(TEST_DATASITE_NAME) / new_fname, data=f.read())
    assert path.exists()


def test_create_permfile(sync_client: SyncClient):
    invalid_contents = b"wrong permfile"
    folder = "test"
    relative_path = Path(TEST_DATASITE_NAME) / folder / PERM_FILE

    # invalid
    with pytest.raises(SyftServerError):
        sync_client.create(relative_path=relative_path, data=invalid_contents)

    # valid
    valid_contents = yaml.safe_dump(
        [
            {
                "path": "a",
                "user": "*",
                "permissions": ["write"],
            }
        ]
    ).encode()
    sync_client.create(relative_path=relative_path, data=valid_contents)


def test_update_permfile_success(sync_client: SyncClient):
    local_data = yaml.safe_dump(
        [
            {
                "path": "a",
                "user": "*",
                "permissions": ["write"],
            }
        ]
    ).encode()

    remote_metadata = sync_client.get_metadata(Path(TEST_DATASITE_NAME) / PERM_FILE)

    diff = py_fast_rsync.diff(remote_metadata.signature_bytes, local_data)
    expected_hash = hashlib.sha256(local_data).hexdigest()

    response = sync_client.apply_diff(Path(TEST_DATASITE_NAME) / PERM_FILE, diff, expected_hash)
    assert isinstance(response, ApplyDiffResponse)


def test_update_permfile_failure(sync_client: SyncClient):
    local_data = (
        b'3gwrehtytrterfewdw ["x@x.org"], "read": ["x@x.org"], "write": ["x@x.org"], "filepath": "~/syftperm.yaml",}'
    )

    remote_metadata = sync_client.get_metadata(Path(TEST_DATASITE_NAME) / PERM_FILE)

    diff = py_fast_rsync.diff(remote_metadata.signature_bytes, local_data)
    expected_hash = hashlib.sha256(local_data).hexdigest()

    with pytest.raises(SyftServerError):
        sync_client.apply_diff(Path(TEST_DATASITE_NAME) / PERM_FILE, diff, expected_hash)


def test_list_datasites(client: TestClient):
    response = client.post("/sync/datasites")

    response.raise_for_status()


def test_get_all_datasite_states(sync_client: SyncClient):
    response = sync_client.get_datasite_states()
    assert len(response) == 1

    metadatas = response[TEST_DATASITE_NAME]
    assert len(metadatas) == 3
    assert all(isinstance(m, FileMetadata) for m in metadatas)


def test_download_snapshot(sync_client: SyncClient, tmpdir: Path):
    tmpdir = Path(tmpdir)
    metadata = sync_client.get_remote_state(Path(TEST_DATASITE_NAME))
    paths = [m.path for m in metadata]
    filelist = sync_client.download_files_streaming(paths, tmpdir)
    assert len(filelist) == 3


def test_whoami(client: TestClient):
    response = client.post("/auth/whoami")
    response.raise_for_status()
    assert response.json() == {"email": TEST_DATASITE_NAME}


def test_large_file_failure(client: TestClient):
    large_data = b"0" * 1024 * 1024 * 11  # 11MB
    response = client.post(
        "/sync/create",
        files={"file": ("large.txt", large_data, "text/plain")},
    )

    assert response.status_code == 413
    assert response.text == "Request Denied. Message size is greater than 10 MB"
