import tempfile
from pathlib import Path

import pytest

from syft_client.sync.peers.peer import Peer
from syft_client.sync.peers.peer_store import PeerStore
from syft_client.sync.syftbox_manager import SyftboxManager
from tests.unit.test_sync_manager import path_for_job
from tests.unit.utils import create_tmp_dataset_files


# =========================================================================
# PeerStore unit tests
# =========================================================================


def test_key_generation():
    ps = PeerStore(email="alice@example.com", use_encryption=True)
    assert not ps.has_my_keys()
    ps.generate_keys()
    assert ps.has_my_keys()
    assert ps.public_key is not None
    assert ps.get_public_bundle() is not None


def test_encrypt_decrypt_roundtrip():
    alice = PeerStore(email="alice@example.com", use_encryption=True)
    alice.generate_keys()
    bob = PeerStore(email="bob@example.com", use_encryption=True)
    bob.generate_keys()

    # Add peers so bundles can be stored on Peer objects
    alice.add_peer(Peer(email="bob@example.com"))
    bob.add_peer(Peer(email="alice@example.com"))

    # Exchange bundles
    alice.set_peer_bundle("bob@example.com", bob.get_public_bundle())
    bob.set_peer_bundle("alice@example.com", alice.get_public_bundle())

    plaintext = b"Hello, this is a secret message!"
    envelope = alice.encrypt("bob@example.com", plaintext)
    assert envelope != plaintext

    decrypted = bob.decrypt("alice@example.com", envelope)
    assert decrypted == plaintext


def test_encrypt_without_keys_raises():
    ps = PeerStore(email="alice@example.com", use_encryption=True)
    ps.add_peer(Peer(email="bob@example.com"))
    with pytest.raises(ValueError, match="No private keys"):
        ps.encrypt("bob@example.com", b"data")


def test_encrypt_without_peer_bundle_raises():
    ps = PeerStore(email="alice@example.com", use_encryption=True)
    ps.generate_keys()
    ps.add_peer(Peer(email="bob@example.com"))
    with pytest.raises(ValueError, match="No public encryption bundle"):
        ps.encrypt("bob@example.com", b"data")


def test_try_decrypt_no_keys():
    ps = PeerStore(email="alice@example.com", use_encryption=True)
    data = b"some unencrypted data"
    with pytest.raises(ValueError, match="No private keys"):
        ps.decrypt("bob@example.com", data) == data


def test_try_decrypt_no_peer_bundle():
    ps = PeerStore(email="alice@example.com", use_encryption=True)
    ps.generate_keys()
    data = b"some unencrypted data"
    with pytest.raises(ValueError, match="No cached peer for"):
        ps.decrypt("bob@example.com", data) == data


def test_try_decrypt_invalid_envelope():
    alice = PeerStore(email="alice@example.com", use_encryption=True)
    alice.generate_keys()
    bob = PeerStore(email="bob@example.com", use_encryption=True)
    bob.generate_keys()

    alice.add_peer(Peer(email="bob@example.com"))
    bob.add_peer(Peer(email="alice@example.com"))

    alice.set_peer_bundle("bob@example.com", bob.get_public_bundle())
    bob.set_peer_bundle("alice@example.com", alice.get_public_bundle())

    bad_data = b"not encrypted at all"
    with pytest.raises(ValueError, match="Serialization error"):
        bob.decrypt("alice@example.com", bad_data)


def test_save_and_load_keys():
    alice = PeerStore(email="alice@example.com", use_encryption=True)
    alice.generate_keys()
    bob = PeerStore(email="bob@example.com", use_encryption=True)
    bob.generate_keys()

    alice.add_peer(Peer(email="bob@example.com"))
    alice.set_peer_bundle("bob@example.com", bob.get_public_bundle())

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = Path(f.name)

    alice.save_keys(path)
    loaded = PeerStore.load_keys(path)

    assert loaded.email == "alice@example.com"
    assert loaded.has_my_keys()
    assert loaded.get_public_bundle() is not None
    assert loaded.has_peer_bundle("bob@example.com")
    path.unlink()


def test_from_keys_data():
    alice = PeerStore(email="alice@example.com", use_encryption=True)
    alice.generate_keys()

    keys_data = {"keys_jwk": alice._private_keys.to_jwks()}
    loaded = PeerStore.from_keys_data("alice@example.com", keys_data)
    assert loaded.has_my_keys()
    assert loaded.get_public_bundle() is not None


def test_peer_use_encryption_inherited():
    """Peers added to PeerStore inherit its use_encryption flag."""
    ps = PeerStore(email="alice@example.com", use_encryption=True)
    peer = Peer(email="bob@example.com")
    assert not peer.use_encryption
    ps.add_peer(peer)
    assert peer.use_encryption

    ps2 = PeerStore(email="alice@example.com", use_encryption=False)
    peer2 = Peer(email="bob@example.com", use_encryption=True)
    ps2.add_peer(peer2)
    assert not peer2.use_encryption


def test_set_peers_inherits_encryption():
    """set_peers propagates use_encryption to all peers."""
    ps = PeerStore(email="alice@example.com", use_encryption=True)
    peers = [Peer(email="bob@example.com"), Peer(email="carol@example.com")]
    ps.set_peers(peers)
    assert all(p.use_encryption for p in peers)


# =========================================================================
# Full DS→DO→DS flow with encryption via mock drive
# =========================================================================


def test_encrypted_message_flow():
    """Full sync flow with encryption: DS sends message, DO receives and decrypts."""
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        encryption=True,
    )

    # Verify peer stores are set with encryption enabled
    assert ds_manager._peer_store is not None
    assert do_manager._peer_store is not None

    # DO has DS's bundle after approval
    do_ps = do_manager._peer_store
    assert do_ps.has_peer_bundle(ds_manager.email)

    # DS needs to load_peers to pick up DO's bundle
    ds_manager.load_peers()
    ds_ps = ds_manager._peer_store
    assert ds_ps.has_peer_bundle(do_manager.email)

    # DS sends a file change (encrypted)
    file_path = path_for_job(do_manager.email, ds_manager.email, "my.job")
    ds_manager._send_file_change(file_path, "encrypted content")

    # DO syncs and should receive the decrypted content
    do_manager.sync()

    events = do_manager._get_all_accepted_events_do()
    assert len(events) > 0

    all_events = [e for msg in events for e in msg.events]
    matching = [e for e in all_events if "encrypted content" in str(e.content)]
    assert len(matching) > 0


def test_encrypted_sync_down_ds():
    """DS can pull encrypted events from DO outbox."""
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        encryption=True,
    )

    # DS needs DO's bundle to decrypt
    ds_manager.load_peers()

    # DS sends a file
    file_path = path_for_job(do_manager.email, ds_manager.email)
    ds_manager._send_file_change(file_path, "test data")

    # DO syncs (receives and processes)
    do_manager.sync()

    # DS syncs down (pulls encrypted events from DO outbox)
    ds_manager.sync()

    # DS should have cached events
    cached = (
        ds_manager.datasite_watcher_syncer.datasite_watcher_cache.get_cached_events()
    )
    assert len(cached) > 0


def test_no_encryption_backward_compat():
    """Without encryption, everything works as before."""
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        encryption=False,
    )

    assert ds_manager._peer_store is None
    assert do_manager._peer_store is None

    file_path = path_for_job(do_manager.email, ds_manager.email, "my.job")
    ds_manager._send_file_change(file_path, "hello unencrypted")
    do_manager.sync()

    events = do_manager._get_all_accepted_events_do()
    assert len(events) > 0


def test_bundle_exchange_through_peer_approval():
    """Bundles are exchanged during add_peer/approve flow."""
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        encryption=True,
        add_peers=False,
    )

    ds_ps = ds_manager._peer_store
    do_ps = do_manager._peer_store

    # Before peer add, no peer bundles
    assert not ds_ps.has_peer_bundle(do_manager.email)
    assert not do_ps.has_peer_bundle(ds_manager.email)

    # DS adds peer (writes bundle)
    ds_manager.add_peer(do_manager.email)

    # DO loads peers and approves (reads DS bundle, writes own)
    do_manager.load_peers()
    do_manager.approve_peer_request(ds_manager.email)

    # DO should now have DS's bundle
    assert do_ps.has_peer_bundle(ds_manager.email)

    # DS loads peers again to pick up DO's bundle
    ds_manager.load_peers()
    assert ds_ps.has_peer_bundle(do_manager.email)


# =========================================================================
# Self-encryption (at-rest) tests
# =========================================================================


def test_self_encrypt_decrypt_roundtrip():
    """PeerStore can encrypt and decrypt data for itself."""
    ps = PeerStore(email="alice@example.com", use_encryption=True)
    ps.generate_keys()

    plaintext = b"secret at-rest data"
    encrypted = ps.encrypt_for_self(plaintext)
    assert encrypted != plaintext

    decrypted = ps.decrypt_for_self(encrypted)
    assert decrypted == plaintext


def test_self_encrypt_if_needed_with_encryption():
    """encrypt_for_self_if_needed encrypts when enabled and keys exist."""
    ps = PeerStore(email="alice@example.com", use_encryption=True)
    ps.generate_keys()

    data = b"hello"
    encrypted = ps.encrypt_for_self_if_needed(data)
    assert encrypted != data

    decrypted = ps.decrypt_and_verify_for_self_if_needed(encrypted)
    assert decrypted == data


def test_self_encrypt_if_needed_without_encryption():
    """encrypt_for_self_if_needed is a no-op when encryption is disabled."""
    ps = PeerStore(email="alice@example.com", use_encryption=False)
    ps.generate_keys()

    data = b"hello"
    assert ps.encrypt_for_self_if_needed(data) == data
    assert ps.decrypt_and_verify_for_self_if_needed(data) == data


def test_self_encrypt_if_needed_without_keys():
    """encrypt_for_self_if_needed is a no-op when keys are missing."""
    ps = PeerStore(email="alice@example.com", use_encryption=True)

    data = b"hello"
    assert ps.encrypt_for_self_if_needed(data) == data
    assert ps.decrypt_and_verify_for_self_if_needed(data) == data


# =========================================================================
# At-rest encryption via mock drive: events, checkpoints, rolling state
# =========================================================================


def test_encrypted_events_at_rest():
    """DO events are encrypted at rest and decrypted on read back."""
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        encryption=True,
    )
    ds_manager.load_peers()

    # DS sends a file change
    file_path = path_for_job(do_manager.email, ds_manager.email, "my.job")
    ds_manager._send_file_change(file_path, "at-rest encrypted")
    do_manager.sync()

    # Read back events — should be decrypted transparently
    events = do_manager._get_all_accepted_events_do()
    assert len(events) > 0
    all_events = [e for msg in events for e in msg.events]
    matching = [e for e in all_events if "at-rest encrypted" in str(e.content)]
    assert len(matching) > 0


def test_encrypted_checkpoint_roundtrip():
    """Checkpoint is encrypted at rest and decrypted on read."""
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        encryption=True,
    )
    ds_manager.load_peers()

    # Create some data and a checkpoint
    file_path = path_for_job(do_manager.email, ds_manager.email, "my.job")
    ds_manager._send_file_change(file_path, "checkpoint data")
    do_manager.sync()

    checkpoint = do_manager.datasite_owner_syncer.create_checkpoint()
    assert len(checkpoint.files) > 0

    # Read back checkpoint — should decrypt transparently
    loaded = do_manager._connection_router.get_latest_checkpoint()
    assert loaded is not None
    assert len(loaded.files) == len(checkpoint.files)


def test_encrypted_rolling_state_roundtrip():
    """Rolling state is encrypted at rest and decrypted on read."""
    from syft_client.sync.checkpoints.rolling_state import RollingState
    from syft_client.sync.events.file_change_event import (
        FileChangeEvent,
        FileChangeEventsMessage,
    )
    from pathlib import Path
    from uuid import uuid4

    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        encryption=True,
    )

    # Create and upload rolling state
    rs = RollingState(email=do_manager.email, base_checkpoint_timestamp=0.0)
    event = FileChangeEvent(
        id=uuid4(),
        path_in_datasite=Path("test/file.txt"),
        content=b"rolling state content",
        old_hash=None,
        new_hash="abc123",
        is_deleted=False,
        submitted_timestamp=1.0,
        timestamp=1.0,
        datasite_email=do_manager.email,
    )
    rs.add_events_message(FileChangeEventsMessage(events=[event]))

    do_manager._connection_router.upload_rolling_state(rs)

    # Read back — should decrypt transparently
    loaded = do_manager._connection_router.get_rolling_state()
    assert loaded is not None
    assert loaded.event_count == 1


def test_at_rest_no_encryption_backward_compat():
    """Events, checkpoints, rolling state work without encryption."""
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        encryption=False,
    )

    file_path = path_for_job(do_manager.email, ds_manager.email, "my.job")
    ds_manager._send_file_change(file_path, "unencrypted data")
    do_manager.sync()

    events = do_manager._get_all_accepted_events_do()
    assert len(events) > 0

    checkpoint = do_manager.datasite_owner_syncer.create_checkpoint()
    assert len(checkpoint.files) > 0

    loaded = do_manager._connection_router.get_latest_checkpoint()
    assert loaded is not None
    assert len(loaded.files) == len(checkpoint.files)


# =========================================================================
# Signature verification tests
# =========================================================================


def _setup_alice_bob():
    """Helper: create alice and bob PeerStores with exchanged bundles."""
    alice = PeerStore(email="alice@example.com", use_encryption=True)
    alice.generate_keys()
    bob = PeerStore(email="bob@example.com", use_encryption=True)
    bob.generate_keys()

    alice.add_peer(Peer(email="bob@example.com"))
    bob.add_peer(Peer(email="alice@example.com"))

    alice.set_peer_bundle("bob@example.com", bob.get_public_bundle())
    bob.set_peer_bundle("alice@example.com", alice.get_public_bundle())
    return alice, bob


def test_verify_message():
    """Verify signature of a message from a known sender."""
    alice, bob = _setup_alice_bob()

    plaintext = b"signed message"
    envelope = alice.encrypt("bob@example.com", plaintext)

    # Bob verifies Alice's signature — should not raise
    bob.verify_message("alice@example.com", envelope)


def test_verify_message_wrong_sender_raises():
    """Verification fails when checked against the wrong sender's key."""
    alice, bob = _setup_alice_bob()

    carol = PeerStore(email="carol@example.com", use_encryption=True)
    carol.generate_keys()
    bob.add_peer(Peer(email="carol@example.com"))
    bob.set_peer_bundle("carol@example.com", carol.get_public_bundle())

    envelope = alice.encrypt("bob@example.com", b"hello")

    with pytest.raises(Exception):
        bob.verify_message("carol@example.com", envelope)


def test_decrypt_tampered_envelope_raises():
    """Decrypting a tampered envelope should raise."""
    alice, bob = _setup_alice_bob()

    envelope = alice.encrypt("bob@example.com", b"secret")
    tampered = bytearray(envelope)
    tampered[-1] ^= 0xFF
    tampered = bytes(tampered)

    with pytest.raises(Exception):
        bob.decrypt("alice@example.com", tampered)


def test_verify_message_from_self():
    """Verify signature of a self-encrypted envelope."""
    ps = PeerStore(email="alice@example.com", use_encryption=True)
    ps.generate_keys()

    plaintext = b"self-signed data"
    envelope = ps.encrypt_for_self(plaintext)

    # Should not raise
    ps.verify_message_from_self(envelope)


# =========================================================================
# E2E signature verification through mock drive sync
# =========================================================================


def test_verified_encrypted_message_flow():
    """Good message passes signature verification + decryption through full sync."""
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        encryption=True,
    )
    ds_manager.load_peers()

    # DS sends a file change (encrypted + signed)
    file_path = path_for_job(do_manager.email, ds_manager.email, "my.job")
    ds_manager._send_file_change(file_path, "verified content")

    # DO syncs — decrypt_and_verify_if_needed runs internally
    do_manager.sync()

    events = do_manager._get_all_accepted_events_do()
    assert len(events) > 0
    all_events = [e for msg in events for e in msg.events]
    matching = [e for e in all_events if "verified content" in str(e.content)]
    assert len(matching) > 0


def test_tampered_inbox_message_raises():
    """Tampered message in mock inbox fails signature verification on sync."""
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        encryption=True,
    )
    ds_manager.load_peers()

    # DS sends a file change
    file_path = path_for_job(do_manager.email, ds_manager.email, "my.job")
    ds_manager._send_file_change(file_path, "will be tampered")

    # Tamper with raw bytes in the mock backing store
    do_conn = do_manager._connection_router.connections[0]
    inbox_id = do_conn._get_own_datasite_inbox_id(ds_manager.email)
    files = do_conn.get_file_metadatas_from_folder(inbox_id)
    assert len(files) > 0

    file_id = files[0]["id"]
    mock_file = do_conn.drive_service._backing_store.files[file_id]
    tampered = bytearray(mock_file.content)
    tampered[-1] ^= 0xFF
    mock_file.content = bytes(tampered)

    # DO sync should fail due to signature verification
    with pytest.raises(Exception):
        do_manager.sync()


def test_encrypted_dataset_collection_syncs():
    """Dataset-collection sync-down works under encryption."""
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        encryption=True,
    )

    mock_path, private_path, readme_path = create_tmp_dataset_files()
    do_manager.create_dataset(
        name="demo",
        mock_path=mock_path,
        private_path=private_path,
        summary="demo dataset",
        readme_path=readme_path,
        users=[ds_manager.email],
        upload_private=True,
        sync=False,
    )
    do_manager.sync()

    ds_manager.sync()

    cr = ds_manager.peer_manager.connection_router
    collections = cr.watcher_list_dataset_collections()
    do_collections = [c for c in collections if c["owner_email"] == do_manager.email]
    assert do_collections, "DS does not see the DO's dataset collection"

    c = do_collections[0]
    files = cr.watcher_download_dataset_collection(
        c["tag"], c["content_hash"], do_manager.email
    )
    assert files, "DS could not download the dataset collection files"
