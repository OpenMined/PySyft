"""Peer public key bundles are validated, pinned locally, and never swapped silently.

The bundle of a peer arrives over Google Drive, which the threat model treats
as an adversarial transport. These tests cover the checks at ingestion (DID
identity + signatures), the local pin that outlives the Drive copy of
SYFT_peers.json, and how a rotated key is reported and then trusted.
"""

import json
import warnings

import pytest

from syft.sync.connections.drive.gdrive_transport import SYFT_PEERS_FILE
from syft.sync.peers.key_bundle import (
    InvalidPeerBundleError,
    PeerKeyChangedError,
    bundle_fingerprint,
    format_fingerprint,
)
from syft.sync.peers.peer import Peer
from syft.sync.peers.peer_store import PeerStore, datasite_crypto_keys_path
from syft.sync.syftbox_manager import SyftboxManager
from tests.unit.test_sync_manager import path_for_job
from tests.unit.utils import grant_job_inbox_access

ALICE = "alice@example.com"
BOB = "bob@example.com"
MALLORY = "mallory@example.com"


def _store(email: str) -> PeerStore:
    store = PeerStore(email=email, use_encryption=True)
    store.generate_keys()
    return store


def _alice_with_bob_pinned() -> tuple[PeerStore, PeerStore]:
    alice, bob = _store(ALICE), _store(BOB)
    alice.add_peer(Peer(email=BOB))
    alice.set_peer_bundle(BOB, bob.get_public_bundle())
    return alice, bob


# =========================================================================
# Validation at ingestion
# =========================================================================


def test_valid_bundle_is_accepted_and_fingerprinted():
    alice, bob = _alice_with_bob_pinned()
    assert alice.peer_fingerprint(BOB) == bob.my_fingerprint
    assert alice.get_cached_peer(BOB).fingerprint == bob.my_fingerprint
    # 64 hex chars in groups of four, readable over the phone.
    assert len(format_fingerprint(bob.my_fingerprint).split(" ")) == 16


def test_bundle_filed_under_another_email_is_refused():
    alice, mallory = _store(ALICE), _store(MALLORY)
    alice.add_peer(Peer(email=BOB))
    # Mallory's real bundle, filed as if it were Bob's.
    with pytest.raises(InvalidPeerBundleError, match="asserts the identity"):
        alice.set_peer_bundle(BOB, mallory.get_public_bundle())
    assert not alice.has_peer_bundle(BOB)


def test_bundle_with_forged_did_but_foreign_identity_field_is_refused():
    alice, mallory = _store(ALICE), _store(MALLORY)
    alice.add_peer(Peer(email=BOB))
    forged = mallory.get_public_bundle()
    forged["id"] = f"did:syft:{BOB}"
    with pytest.raises(InvalidPeerBundleError, match="carries identity"):
        alice.set_peer_bundle(BOB, forged)


def test_did_email_comparison_ignores_case():
    alice, bob = _store(ALICE), _store(BOB)
    alice.add_peer(Peer(email=BOB.upper()))
    alice.set_peer_bundle(BOB.upper(), bob.get_public_bundle())
    assert alice.has_peer_bundle(BOB.upper())


def test_bundle_with_swapped_key_material_is_refused():
    alice, bob, mallory = _store(ALICE), _store(BOB), _store(MALLORY)
    alice.add_peer(Peer(email=BOB))
    tampered = bob.get_public_bundle()
    # Keep Bob's DID and identity key, swap in Mallory's key-agreement keys.
    tampered["keyAgreement"] = mallory.get_public_bundle()["keyAgreement"]
    with pytest.raises(InvalidPeerBundleError, match="parse|signature"):
        alice.set_peer_bundle(BOB, tampered)


def test_pinned_bundle_that_no_longer_validates_is_refused_at_use():
    alice, bob = _alice_with_bob_pinned()
    # Corrupt the pin behind the store's back (as a bad key file would).
    alice.get_cached_peer(BOB).public_encryption_bundle["id"] = f"did:syft:{MALLORY}"
    with pytest.raises(InvalidPeerBundleError):
        alice.encrypt(BOB, b"hello")


# =========================================================================
# Pinning
# =========================================================================


def test_changed_key_is_refused_unless_change_is_allowed():
    alice, bob = _alice_with_bob_pinned()
    old_fp = alice.peer_fingerprint(BOB)

    new_bob = _store(BOB)  # Bob reinstalled: same email, new keys
    with pytest.raises(PeerKeyChangedError) as info:
        alice.set_peer_bundle(BOB, new_bob.get_public_bundle())
    assert info.value.old_fingerprint == old_fp
    assert info.value.new_fingerprint == new_bob.my_fingerprint
    assert alice.peer_fingerprint(BOB) == old_fp

    alice.set_peer_bundle(BOB, new_bob.get_public_bundle(), allow_key_change=True)
    assert alice.peer_fingerprint(BOB) == new_bob.my_fingerprint


def test_re_pinning_same_key_is_no_op():
    alice, bob = _alice_with_bob_pinned()
    alice.set_peer_bundle(BOB, bob.get_public_bundle())
    assert alice.peer_fingerprint(BOB) == bob.my_fingerprint


def test_set_peers_keeps_pin_over_differing_cached_bundle():
    """What load_peers does: the peer list is rebuilt from SYFT_peers.json."""
    alice, bob = _alice_with_bob_pinned()
    mallory = _store(MALLORY)
    swapped = mallory.get_public_bundle()
    swapped["id"] = f"did:syft:{BOB}"
    swapped["identity"] = BOB

    with pytest.warns(UserWarning, match="differs from the pinned key"):
        alice.set_peers([Peer(email=BOB, public_encryption_bundle=swapped)])

    assert alice.peer_fingerprint(BOB) == bob.my_fingerprint


def test_set_peers_adopts_re_signed_prekeys_under_same_identity_key():
    """The identity key is what is pinned; prekeys it signed may change freely."""
    alice, bob = _alice_with_bob_pinned()
    refreshed = bob.get_public_bundle()
    refreshed["@context"] = list(refreshed["@context"]) + ["https://example.com/x"]
    assert refreshed != alice.get_cached_peer(BOB).public_encryption_bundle

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        alice.set_peers([Peer(email=BOB, public_encryption_bundle=refreshed)])
    assert alice.get_cached_peer(BOB).public_encryption_bundle == refreshed


def test_set_peers_pins_valid_cached_bundle_on_first_use():
    alice, bob = _store(ALICE), _store(BOB)
    alice.set_peers([Peer(email=BOB, public_encryption_bundle=bob.get_public_bundle())])
    assert alice.peer_fingerprint(BOB) == bob.my_fingerprint


def test_set_peers_drops_invalid_cached_bundle():
    alice, mallory = _store(ALICE), _store(MALLORY)
    with pytest.warns(UserWarning, match="Dropping cached encryption key"):
        alice.set_peers(
            [Peer(email=BOB, public_encryption_bundle=mallory.get_public_bundle())]
        )
    assert not alice.has_peer_bundle(BOB)


# =========================================================================
# Local persistence of pins
# =========================================================================


def test_pins_persist_in_private_key_file(tmp_path):
    path = tmp_path / "crypto_keys.json"
    alice = PeerStore.create(email=ALICE, use_encryption=True, keys_path=path)
    bob = _store(BOB)
    alice.add_peer(Peer(email=BOB))
    alice.set_peer_bundle(BOB, bob.get_public_bundle())

    on_disk = json.loads(path.read_text())["peer_bundles"]
    assert bundle_fingerprint(on_disk[BOB]) == bob.my_fingerprint

    reloaded = PeerStore.create(email=ALICE, use_encryption=True, keys_path=path)
    assert reloaded.peer_fingerprint(BOB) == bob.my_fingerprint


def test_persisting_pin_keeps_pins_written_by_another_process(tmp_path):
    path = tmp_path / "crypto_keys.json"
    alice = PeerStore.create(email=ALICE, use_encryption=True, keys_path=path)
    bob, carol = _store(BOB), _store("carol@example.com")

    # Another process (syft-bg) pinned Carol meanwhile.
    data = json.loads(path.read_text())
    data["peer_bundles"] = {carol.email: carol.get_public_bundle()}
    path.write_text(json.dumps(data))

    alice.add_peer(Peer(email=BOB))
    alice.set_peer_bundle(BOB, bob.get_public_bundle())

    on_disk = json.loads(path.read_text())["peer_bundles"]
    assert set(on_disk) == {BOB, carol.email}


def test_key_trusted_by_another_process_is_adopted_without_warning(tmp_path):
    """A notebook and syft-bg share one key file; whichever trusted a new key wins."""
    path = tmp_path / "crypto_keys.json"
    notebook = PeerStore.create(email=ALICE, use_encryption=True, keys_path=path)
    bob = _store(BOB)
    notebook.add_peer(Peer(email=BOB))
    notebook.set_peer_bundle(BOB, bob.get_public_bundle())

    new_bob = _store(BOB)
    daemon = PeerStore.create(email=ALICE, use_encryption=True, keys_path=path)
    daemon.set_peer_bundle(BOB, new_bob.get_public_bundle(), allow_key_change=True)

    # The notebook reloads peers from SYFT_peers.json, which the daemon updated.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        notebook.set_peers(
            [Peer(email=BOB, public_encryption_bundle=new_bob.get_public_bundle())]
        )
    assert notebook.peer_fingerprint(BOB) == new_bob.my_fingerprint


def test_corrupt_stored_pin_is_dropped_on_load(tmp_path):
    path = tmp_path / "crypto_keys.json"
    alice = PeerStore.create(email=ALICE, use_encryption=True, keys_path=path)
    bob = _store(BOB)
    alice.add_peer(Peer(email=BOB))
    alice.set_peer_bundle(BOB, bob.get_public_bundle())

    data = json.loads(path.read_text())
    data["peer_bundles"][BOB]["id"] = f"did:syft:{MALLORY}"
    path.write_text(json.dumps(data))

    reloaded = PeerStore.load_keys(path)
    assert not reloaded.has_peer_bundle(BOB)


# =========================================================================
# Through the manager, over the mock Drive
# =========================================================================


def _backing_store(manager: SyftboxManager):
    connection = manager.peer_manager.connection_router.connections[0]
    return connection.drive_service._backing_store


def _peers_json_file(manager: SyftboxManager):
    files = [
        f
        for f in _backing_store(manager).files.values()
        if f.name == SYFT_PEERS_FILE
        and f.owners
        and f.owners[0]["emailAddress"] == manager.email
    ]
    assert len(files) == 1
    return files[0]


def _bundle_file_of(manager: SyftboxManager, for_email: str):
    name = f"encryption_bundle_{manager.email}_for_{for_email}.json"
    files = [f for f in _backing_store(manager).files.values() if f.name == name]
    assert len(files) == 1
    return files[0]


def test_swapped_bundle_in_drive_peers_file_is_not_adopted():
    ds, do = SyftboxManager.pair_with_mock_drive_service_connection(encryption=True)
    grant_job_inbox_access(do, ds.email)
    ds.load_peers()
    original = do.peer_fingerprint(ds.email)
    assert original == ds.encryption_fingerprint

    # Someone with write access to Drive bytes swaps the DS entry in the DO's
    # SYFT_peers.json for a key they control, consistently signed and filed
    # under the DS's DID.
    mallory = _store(MALLORY)
    swapped = mallory.get_public_bundle()
    swapped["id"] = f"did:syft:{ds.email}"
    swapped["identity"] = ds.email
    peers_file = _peers_json_file(do)
    data = json.loads(peers_file.content)
    data[ds.email]["public_encryption_bundle"] = swapped
    peers_file.content = json.dumps(data).encode()

    with pytest.warns(UserWarning, match="differs from the pinned key"):
        do.load_peers(force_download=True)

    assert do.peer_fingerprint(ds.email) == original

    # Encrypted traffic still verifies against the real DS key.
    ds._send_file_change(path_for_job(do.email, ds.email, "my.job"), "secret")
    do.sync()
    events = [e for m in do._get_all_accepted_events_do() for e in m.events]
    assert any("secret" in str(e.content) for e in events)


def test_rotated_peer_key_is_reported_on_load_and_adopted_on_trust():
    ds, do = SyftboxManager.pair_with_mock_drive_service_connection(encryption=True)
    grant_job_inbox_access(do, ds.email)
    ds.load_peers()
    old_fp = do.peer_fingerprint(ds.email)

    # The DS reinstalls: new keys, and re-publishes the bundle for the DO.
    ds._init_encrypted_peer_store()
    ds.peer_manager._write_encryption_bundle_for_peer(do.email)
    new_fp = ds.encryption_fingerprint
    assert new_fp != old_fp

    with pytest.warns(UserWarning, match="differs from\\s+the pinned key"):
        do.load_peers(force_download=True)
    assert do.peer_fingerprint(ds.email) == old_fp

    with pytest.warns(UserWarning, match="Trusting the new key"):
        assert do.trust_peer_key(ds.email) == new_fp
    assert do.peer_fingerprint(ds.email) == new_fp

    # The Drive cache now carries the trusted key too.
    data = json.loads(_peers_json_file(do).content)
    assert bundle_fingerprint(data[ds.email]["public_encryption_bundle"]) == new_fp


def test_approving_peer_again_after_key_change_warns_and_adopts():
    ds, do = SyftboxManager.pair_with_mock_drive_service_connection(encryption=True)
    old_fp = do.peer_fingerprint(ds.email)

    ds._init_encrypted_peer_store()
    ds.peer_manager._write_encryption_bundle_for_peer(do.email)

    with pytest.warns(UserWarning, match="Trusting the new key"):
        do.add_peer(ds.email, force=True, sync=False)
    assert do.peer_fingerprint(ds.email) == ds.encryption_fingerprint != old_fp


def test_bundle_file_naming_another_identity_is_refused_at_approval():
    ds, do = SyftboxManager.pair_with_mock_drive_service_connection(
        encryption=True, add_peers=False
    )
    ds.add_peer(do.email, sync=False)
    # Replace what the DS published with Mallory's bundle, DID and all.
    mallory = _store(MALLORY)
    bundle_file = _bundle_file_of(ds, do.email)
    bundle_file.content = json.dumps(
        {"public_encryption_bundle": mallory.get_public_bundle()}
    ).encode()

    do.load_peers()
    with pytest.warns(UserWarning, match="Refusing the encryption key"):
        do.approve_peer_request(ds.email)
    assert do.peer_fingerprint(ds.email) is None


def test_pins_survive_new_manager_on_same_datasite():
    ds, do = SyftboxManager.pair_with_mock_drive_service_connection(encryption=True)
    ds.load_peers()
    # Point the DO's store at a key file and pin through it, as a real login does.
    keys_path = datasite_crypto_keys_path(do.syftbox_folder, do.email)
    do._peer_store.save_keys(keys_path)
    do.trust_peer_key(ds.email)

    reloaded = PeerStore.create(
        email=do.email, use_encryption=True, keys_path=keys_path
    )
    assert reloaded.peer_fingerprint(ds.email) == ds.encryption_fingerprint


def test_fingerprints_are_none_without_encryption():
    ds, do = SyftboxManager.pair_with_mock_drive_service_connection(encryption=False)
    assert do.encryption_fingerprint is None
    assert do.peer_fingerprint(ds.email) is None
