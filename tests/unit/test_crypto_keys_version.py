"""The crypto key file carries a version, and any other version stops the load.

A user cannot rebuild a private key, so delete-and-rebuild is not a recovery
here. A client must refuse a file of another version rather than read it wrong
and lose the keys.
"""

import json

import pytest
from syft.sync.peers.peer_store import CRYPTO_KEYS_VERSION, PeerStore

EMAIL = "alice@example.com"


def _saved(tmp_path):
    path = tmp_path / "crypto_keys.json"
    PeerStore.create(email=EMAIL, keys_path=path)
    return path


def _set_version(path, version):
    data = json.loads(path.read_text())
    data["version"] = version
    path.write_text(json.dumps(data))


def test_a_saved_file_carries_the_version(tmp_path):
    data = json.loads(_saved(tmp_path).read_text())
    assert data["version"] == CRYPTO_KEYS_VERSION


def test_a_saved_file_loads_back(tmp_path):
    path = _saved(tmp_path)
    on_disk = path.read_text()
    PeerStore.create(email=EMAIL, keys_path=path)
    assert path.read_text() == on_disk


@pytest.mark.parametrize(
    "version",
    [CRYPTO_KEYS_VERSION - 1, CRYPTO_KEYS_VERSION + 1],
    ids=["older", "newer"],
)
def test_a_file_of_another_version_is_refused(tmp_path, version):
    path = _saved(tmp_path)
    _set_version(path, version)

    with pytest.raises(ValueError, match=f"version {version}"):
        PeerStore.create(email=EMAIL, keys_path=path)


@pytest.mark.parametrize(
    "change",
    [lambda d: d.pop("keys_jwk"), lambda d: d.update(peer_bundles={})],
    ids=["missing field", "extra field"],
)
def test_a_file_of_another_shape_is_refused(tmp_path, change):
    path = _saved(tmp_path)
    data = json.loads(path.read_text())
    change(data)
    path.write_text(json.dumps(data))

    with pytest.raises(ValueError, match="_KeyFile"):
        PeerStore.create(email=EMAIL, keys_path=path)


def test_a_file_of_another_identity_is_refused(tmp_path):
    path = _saved(tmp_path)

    with pytest.raises(ValueError, match="belongs to"):
        PeerStore.create(email="bob@example.com", keys_path=path)


def test_saving_never_replaces_an_existing_file(tmp_path):
    """A second process that generated keys at the same time must not win."""
    path = _saved(tmp_path)
    on_disk = path.read_text()
    other = PeerStore(email=EMAIL)
    other.generate_keys()

    with pytest.raises(FileExistsError):
        other.save_keys(path)
    assert path.read_text() == on_disk
    assert [p.name for p in tmp_path.iterdir()] == ["crypto_keys.json"]
