"""The crypto key file carries a version, and an unknown one stops the load.

A user cannot rebuild a private key, so delete-and-rebuild is not a recovery
here. If a newer client wrote the file, this client must refuse it rather than
read it wrong and lose the keys.
"""

import json

import pytest
from syft_client.sync.peers.peer_store import CRYPTO_KEYS_VERSION, PeerStore


def _saved(tmp_path):
    store = PeerStore(email="alice@example.com", use_encryption=True)
    store.generate_keys()
    path = tmp_path / "crypto_keys.json"
    store.save_keys(path)
    return path


def test_a_saved_file_carries_the_version(tmp_path):
    data = json.loads(_saved(tmp_path).read_text())
    assert data["version"] == CRYPTO_KEYS_VERSION


def test_a_saved_file_loads_back(tmp_path):
    path = _saved(tmp_path)
    loaded = PeerStore.load_keys(path)
    assert loaded.email == "alice@example.com"


def test_a_file_without_a_version_still_loads(tmp_path):
    # Written before the version field existed. Those keys must keep working.
    path = _saved(tmp_path)
    data = json.loads(path.read_text())
    del data["version"]
    path.write_text(json.dumps(data))

    loaded = PeerStore.load_keys(path)
    assert loaded.email == "alice@example.com"


def test_a_file_from_a_newer_client_is_refused(tmp_path):
    path = _saved(tmp_path)
    data = json.loads(path.read_text())
    data["version"] = CRYPTO_KEYS_VERSION + 1
    path.write_text(json.dumps(data))

    with pytest.raises(ValueError, match=str(CRYPTO_KEYS_VERSION + 1)):
        PeerStore.load_keys(path)
