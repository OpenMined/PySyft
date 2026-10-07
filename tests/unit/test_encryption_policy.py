"""Per-peer encryption negotiated from each side's advertised policy.

A client is ``optional`` (encrypt only when the peer requires it) or
``required``. A pair encrypts when either side requires it.
"""

import json
import threading
import warnings
from pathlib import Path

import pytest

from syft.sync.connections.connection_router import ConnectionRouter
from syft.sync.connections.drive import mock_drive_service
from syft.sync.connections.drive.gdrive_transport import GDriveConnection
from syft.sync.encryption_policy import (
    EncryptionPolicy,
    EncryptionPolicyConflictError,
    PeerEncryptionPolicyUnknownError,
    PeerKeysNotReadyError,
)
from syft.sync.events.file_change_event import FileChangeEventsMessage
from syft.sync.peers.peer import Peer, PeerState
from syft.sync.peers.peer_store import PeerStore
from syft.sync.sync.datasite_watcher_syncer import DatasiteWatcherSyncer
from syft.sync.syftbox_manager import SyftboxManager
from syft.sync.version.version_info import VersionInfo
from tests.unit.test_sync_manager import path_for_job
from tests.unit.utils import grant_job_inbox_access

OFF = EncryptionPolicy.OFF
OPTIONAL = EncryptionPolicy.OPTIONAL
REQUIRED = EncryptionPolicy.REQUIRED


def _version_with_policy(policy: EncryptionPolicy | None) -> VersionInfo:
    version = VersionInfo.current()
    if policy is not None:
        version.encryption_policy = policy
    return version


def _store_with_peer(
    my_policy: EncryptionPolicy, peer_policy: EncryptionPolicy | None
) -> PeerStore:
    store = PeerStore(email="alice@example.com", encryption_policy=my_policy)
    store.add_peer(
        Peer(email="bob@example.com", version=_version_with_policy(peer_policy))
    )
    return store


def _keyed_pair(
    alice_policy: EncryptionPolicy, bob_policy: EncryptionPolicy
) -> tuple[PeerStore, PeerStore]:
    """Two stores that know each other's policy and hold each other's bundle."""
    alice = PeerStore(email="alice@example.com", encryption_policy=alice_policy)
    bob = PeerStore(email="bob@example.com", encryption_policy=bob_policy)
    alice.generate_keys()
    bob.generate_keys()
    alice.add_peer(
        Peer(email="bob@example.com", version=_version_with_policy(bob_policy))
    )
    bob.add_peer(
        Peer(email="alice@example.com", version=_version_with_policy(alice_policy))
    )
    alice.set_peer_bundle("bob@example.com", bob.get_public_bundle())
    bob.set_peer_bundle("alice@example.com", alice.get_public_bundle())
    return alice, bob


def _raw_inbox_message(do_manager, sender_email: str) -> bytes:
    connection = do_manager.datasite_owner_syncer.connection_router
    raw, _ = (
        connection.connection_for_receive_message().owner_download_next_raw_proposed_message_from_inbox(
            sender_email
        )
    )
    return raw


def _is_envelope(data: bytes) -> bool:
    return PeerStore.is_envelope(data)


def _accepted_contents(do_manager) -> list[str]:
    return [
        str(e.content)
        for msg in do_manager._get_all_accepted_events_do()
        for e in msg.events
    ]


def _ds_cached_contents(ds_manager) -> list[str]:
    cache = ds_manager.datasite_watcher_syncer.datasite_watcher_cache
    return [str(e.content) for e in cache.get_cached_events()]


def _pair_with_stale_sender_view():
    """A required DO and an optional DS whose cached DO policy is outdated."""
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        ds_encryption_policy=OPTIONAL,
        do_encryption_policy=REQUIRED,
    )
    grant_job_inbox_access(do_manager, ds_manager.email)
    ds_manager.load_peers()
    do_manager.load_peers()
    stale = ds_manager.peer_manager.get_cached_peer(do_manager.email)
    stale.version = _version_with_policy(OPTIONAL)
    return ds_manager, do_manager


def _required_ds_pair():
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        ds_encryption_policy=REQUIRED,
        do_encryption_policy=OPTIONAL,
    )
    grant_job_inbox_access(do_manager, ds_manager.email)
    do_manager.load_peers()
    return ds_manager, do_manager


def _owner_sends_with_stale_view(ds_manager, do_manager, content: str) -> None:
    """The owner handles a proposal and answers while it sees the DS as optional."""
    stale = do_manager.peer_manager.get_cached_peer(ds_manager.email)
    stale.version = _version_with_policy(OPTIONAL)
    file_path = path_for_job(do_manager.email, ds_manager.email, "my.job")
    ds_manager._send_file_change(file_path, content)
    do_manager.datasite_owner_syncer.sync([ds_manager.email])


# =========================================================================
# Policy model and advertisement
# =========================================================================


def test_policy_defaults_to_optional():
    assert PeerStore(email="alice@example.com").encryption_policy == OPTIONAL


@pytest.mark.parametrize(
    "use_encryption, expected", [(True, REQUIRED), (False, OPTIONAL)]
)
def test_use_encryption_bool_maps_to_policy(use_encryption, expected):
    store = PeerStore(email="alice@example.com", use_encryption=use_encryption)
    assert store.encryption_policy == expected


@pytest.mark.parametrize(
    "value, expected",
    [(True, REQUIRED), (False, OPTIONAL), ("off", OFF), (OFF, OFF)],
)
def test_encryption_setting_is_coerced_to_policy(value, expected):
    assert EncryptionPolicy.coerce(value) == expected


def test_version_info_without_policy_field_has_no_policy():
    """A version file from a client that predates the field says nothing."""
    data = VersionInfo.current().model_dump(mode="json")
    data.pop("encryption_policy")
    assert VersionInfo.model_validate(data).encryption_policy is None


def _advertise_as_legacy_client(manager) -> None:
    """Write the version file of ``manager`` without the policy field."""
    legacy = manager.peer_manager.get_own_version().model_copy()
    legacy.encryption_policy = None
    manager.peer_manager.connection_router.write_version_file(legacy)


@pytest.mark.parametrize("do_policy, expected", [(REQUIRED, REQUIRED), (OPTIONAL, OFF)])
def test_legacy_peer_policy_follows_its_published_bundle(do_policy, expected):
    """An earlier client publishes a bundle for us only when it encrypts.

    One that does not has no keys and cannot decrypt, so it is off.
    """
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        do_encryption_policy=do_policy,
    )
    _advertise_as_legacy_client(do_manager)
    ds_manager.load_peers()
    assert ds_manager.peer_manager.peer_store.peer_policy(do_manager.email) == expected


def test_own_version_advertises_policy():
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        ds_encryption_policy=OPTIONAL,
        do_encryption_policy=REQUIRED,
    )
    assert do_manager.peer_manager.get_own_version().encryption_policy == REQUIRED
    assert ds_manager.peer_manager.get_own_version().encryption_policy == OPTIONAL
    advertised = ds_manager.peer_manager.get_peer_version(do_manager.email)
    assert advertised.encryption_policy == REQUIRED


def test_missing_version_file_is_created_with_own_policy():
    a_conn, b_conn = mock_drive_service.pair_with_mock_service(
        "a@example.com", "b@example.com"
    )
    a_conn.share_version_file_with_peer("b@example.com", _version_with_policy(REQUIRED))
    assert b_conn.read_peer_version_file("a@example.com").encryption_policy == REQUIRED


def test_policy_change_in_memory_is_republished():
    """A key file marked required makes this client required: peers must see it."""
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        add_peers=False,
        do_encryption_policy=REQUIRED,
    )
    ds_manager.add_peer(do_manager.email, sync=False)
    do_manager.load_peers()
    do_manager.approve_peer_request(ds_manager.email)

    # Another process on the DS datasite wrote a required key file.
    keys_path = ds_manager.peer_manager.peer_store._keys_path
    PeerStore.create(
        email=ds_manager.email, encryption_policy=REQUIRED, keys_path=keys_path
    )
    ds_manager.load_peers()

    assert ds_manager.peer_manager.peer_store.encryption_policy == REQUIRED
    advertised = do_manager.peer_manager.load_peer_version(ds_manager.email)
    assert advertised.encryption_policy == REQUIRED


def test_legacy_policy_is_read_once_per_peer(monkeypatch):
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection()
    _advertise_as_legacy_client(do_manager)
    reads = []
    original = ConnectionRouter.read_peer_encryption_bundle

    def spy(self, email):
        reads.append(email)
        return original(self, email)

    monkeypatch.setattr(ConnectionRouter, "read_peer_encryption_bundle", spy)
    ds_manager.load_peers()
    ds_manager.load_peers()
    assert reads == [do_manager.email]


# =========================================================================
# Per-peer resolution
# =========================================================================


@pytest.mark.parametrize(
    "mine, theirs, encrypted",
    [
        (OPTIONAL, OPTIONAL, False),
        (OPTIONAL, REQUIRED, True),
        (REQUIRED, OPTIONAL, True),
        (REQUIRED, REQUIRED, True),
        (OFF, OFF, False),
        (OFF, OPTIONAL, False),
        (OPTIONAL, OFF, False),
    ],
)
def test_peer_encryption_is_resolved_from_both_policies(mine, theirs, encrypted):
    store = _store_with_peer(mine, theirs)
    assert store.peer_uses_encryption("bob@example.com") is encrypted


@pytest.mark.parametrize(
    "mine, theirs", [(OFF, REQUIRED), (REQUIRED, OFF)], ids=["off", "required"]
)
def test_off_and_required_conflict(mine, theirs):
    """Neither side gives way, so the pair cannot exchange messages."""
    alice, bob = _keyed_pair(mine, theirs)
    assert alice.peer_conflicts("bob@example.com")
    with pytest.raises(EncryptionPolicyConflictError):
        alice.encrypt_if_needed("bob@example.com", b"data")
    with pytest.raises(EncryptionPolicyConflictError):
        alice.decrypt_and_verify_if_needed("bob@example.com", b"plain")


def test_resolution_follows_a_changed_peer_policy():
    """An enclave that reboots with another setting is picked up on the next load."""
    store = _store_with_peer(OPTIONAL, OPTIONAL)
    assert not store.peer_uses_encryption("bob@example.com")

    store.get_cached_peer("bob@example.com").version = _version_with_policy(REQUIRED)
    assert store.peer_uses_encryption("bob@example.com")

    store.get_cached_peer("bob@example.com").version = _version_with_policy(OPTIONAL)
    assert not store.peer_uses_encryption("bob@example.com")


def test_optional_sender_refuses_peer_with_unknown_policy():
    store = PeerStore(email="alice@example.com", encryption_policy=OPTIONAL)
    store.add_peer(Peer(email="bob@example.com"))
    with pytest.raises(PeerEncryptionPolicyUnknownError):
        store.encrypt_if_needed("bob@example.com", b"data")


def test_required_sender_needs_no_peer_policy():
    alice, _ = _keyed_pair(REQUIRED, OPTIONAL)
    alice.get_cached_peer("bob@example.com").version = None
    assert _is_envelope(alice.encrypt_if_needed("bob@example.com", b"data"))


@pytest.mark.parametrize(
    "mine, theirs, with_keys",
    [(OPTIONAL, REQUIRED, False), (REQUIRED, OPTIONAL, True)],
    ids=["no own keys", "no peer bundle"],
)
def test_encrypting_pair_without_keys_or_pin_is_not_ready(mine, theirs, with_keys):
    store = _store_with_peer(mine, theirs)
    if with_keys:
        store.generate_keys()
    with pytest.raises(PeerKeysNotReadyError):
        store.encrypt_if_needed("bob@example.com", b"data")


# =========================================================================
# Keys and pins
# =========================================================================


def test_optional_store_creates_no_key_file(tmp_path):
    keys_path = tmp_path / "crypto_keys.json"
    store = PeerStore.create(
        email="alice@example.com", encryption_policy=OPTIONAL, keys_path=keys_path
    )
    assert not store.has_my_keys()
    assert not keys_path.exists()


def test_off_store_creates_no_key_file(tmp_path):
    keys_path = tmp_path / "crypto_keys.json"
    store = PeerStore.create(
        email="alice@example.com", encryption_policy=OFF, keys_path=keys_path
    )
    assert not store.has_my_keys()
    assert not keys_path.exists()


def test_off_is_refused_when_key_file_requires_encryption(tmp_path):
    """The files at rest of such a datasite are encrypted; off cannot read them."""
    keys_path = tmp_path / "crypto_keys.json"
    with pytest.warns(UserWarning):
        PeerStore.create(
            email="alice@example.com", encryption_policy=REQUIRED, keys_path=keys_path
        )
    with pytest.raises(ValueError, match="new SyftBox folder"):
        PeerStore.create(
            email="alice@example.com", encryption_policy=OFF, keys_path=keys_path
        )


def test_off_keeps_its_policy_over_optional_key_file(tmp_path):
    keys_path = tmp_path / "crypto_keys.json"
    PeerStore.create(email="alice@example.com", keys_path=keys_path).ensure_keys()
    store = PeerStore.create(
        email="alice@example.com", encryption_policy=OFF, keys_path=keys_path
    )
    assert store.encryption_policy == OFF


def test_optional_store_creates_keys_when_needed(tmp_path):
    keys_path = tmp_path / "crypto_keys.json"
    store = PeerStore.create(
        email="alice@example.com", encryption_policy=OPTIONAL, keys_path=keys_path
    )
    store.ensure_keys()
    assert store.has_my_keys()

    reloaded = PeerStore.create(email="alice@example.com", keys_path=keys_path)
    assert reloaded.has_my_keys()
    assert reloaded.encryption_policy == OPTIONAL


def test_optional_store_loads_existing_keys_when_needed(tmp_path):
    keys_path = tmp_path / "crypto_keys.json"
    first = PeerStore.create(
        email="alice@example.com", encryption_policy=REQUIRED, keys_path=keys_path
    )
    second = PeerStore.create(
        email="alice@example.com", encryption_policy=OPTIONAL, keys_path=keys_path
    )
    second.ensure_keys()
    assert second.get_public_bundle() == first.get_public_bundle()


def test_required_policy_in_key_file_wins(tmp_path):
    """syft-bg next to a required notebook loads the same key file and requires too."""
    keys_path = tmp_path / "crypto_keys.json"
    PeerStore.create(
        email="alice@example.com", encryption_policy=REQUIRED, keys_path=keys_path
    )
    daemon = PeerStore.create(
        email="alice@example.com", encryption_policy=OPTIONAL, keys_path=keys_path
    )
    assert daemon.encryption_policy == REQUIRED


def test_key_file_without_policy_reads_as_required(tmp_path):
    """Before the field, only required clients wrote a key file."""
    keys_path = tmp_path / "crypto_keys.json"
    PeerStore.create(
        email="alice@example.com", encryption_policy=REQUIRED, keys_path=keys_path
    )
    data = json.loads(keys_path.read_text())
    del data["encryption_policy"]
    keys_path.write_text(json.dumps(data))

    store = PeerStore.create(email="alice@example.com", keys_path=keys_path)
    assert store.encryption_policy == REQUIRED


def test_required_client_records_policy_in_existing_key_file(tmp_path):
    keys_path = tmp_path / "crypto_keys.json"
    PeerStore.create(email="alice@example.com", keys_path=keys_path).ensure_keys()
    PeerStore.create(
        email="alice@example.com", encryption_policy=REQUIRED, keys_path=keys_path
    )
    store = PeerStore.create(email="alice@example.com", keys_path=keys_path)
    assert store.encryption_policy == REQUIRED


def test_new_required_key_file_checks_files_at_rest_once(tmp_path):
    """An optional client before it wrote plaintext at rest without a key file."""
    keys_path = tmp_path / "crypto_keys.json"
    PeerStore.create(
        email="alice@example.com", encryption_policy=REQUIRED, keys_path=keys_path
    )
    store = PeerStore.create(email="alice@example.com", keys_path=keys_path)
    assert store.needs_at_rest_encryption()


def test_switch_to_required_needs_at_rest_encryption_until_marked(tmp_path):
    keys_path = tmp_path / "crypto_keys.json"
    PeerStore.create(email="alice@example.com", keys_path=keys_path).ensure_keys()
    store = PeerStore.create(
        email="alice@example.com", encryption_policy=REQUIRED, keys_path=keys_path
    )
    assert store.needs_at_rest_encryption()

    store.encrypt_files_at_rest_once(lambda: None)
    reloaded = PeerStore.create(email="alice@example.com", keys_path=keys_path)
    assert not reloaded.needs_at_rest_encryption()


def test_legacy_key_file_checks_files_at_rest_once(tmp_path):
    """An earlier client without encryption can have written plaintext at rest."""
    keys_path = tmp_path / "crypto_keys.json"
    PeerStore.create(
        email="alice@example.com", encryption_policy=REQUIRED, keys_path=keys_path
    )
    data = json.loads(keys_path.read_text())
    del data["encryption_policy"]
    data.pop("at_rest_encrypted", None)
    keys_path.write_text(json.dumps(data))

    store = PeerStore.create(email="alice@example.com", keys_path=keys_path)
    assert store.needs_at_rest_encryption()


def test_first_required_login_warns_about_the_key_file(tmp_path):
    keys_path = tmp_path / "crypto_keys.json"
    with pytest.warns(UserWarning, match="keep a copy"):
        PeerStore.create(
            email="alice@example.com", encryption_policy=REQUIRED, keys_path=keys_path
        )


def test_switch_to_required_warns_about_the_key_file(tmp_path):
    keys_path = tmp_path / "crypto_keys.json"
    PeerStore.create(email="alice@example.com", keys_path=keys_path).ensure_keys()
    with pytest.warns(UserWarning, match="keep a copy"):
        PeerStore.create(
            email="alice@example.com", encryption_policy=REQUIRED, keys_path=keys_path
        )


@pytest.mark.parametrize("policy", [REQUIRED, OPTIONAL])
def test_later_logins_do_not_warn(tmp_path, policy):
    keys_path = tmp_path / "crypto_keys.json"
    with pytest.warns(UserWarning):
        PeerStore.create(
            email="alice@example.com", encryption_policy=REQUIRED, keys_path=keys_path
        )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        PeerStore.create(
            email="alice@example.com", encryption_policy=policy, keys_path=keys_path
        )


def test_optional_keys_do_not_warn(tmp_path):
    keys_path = tmp_path / "crypto_keys.json"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        PeerStore.create(email="alice@example.com", keys_path=keys_path).ensure_keys()


def test_concurrent_processes_make_one_key_pair(tmp_path):
    """A notebook and syft-bg that need keys at once must end with the same keys."""
    keys_path = tmp_path / "crypto_keys.json"
    stores = [
        PeerStore.create(email="alice@example.com", keys_path=keys_path)
        for _ in range(8)
    ]
    barrier = threading.Barrier(len(stores))

    def make_keys(store: PeerStore) -> None:
        barrier.wait()
        store.ensure_keys()

    threads = [threading.Thread(target=make_keys, args=(s,)) for s in stores]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    on_disk = PeerStore.load_keys(keys_path).get_public_bundle()
    assert all(store.get_public_bundle() == on_disk for store in stores)


def test_optional_store_without_keys_ignores_drive_bundle():
    """Without keys there is no pin, so a bundle from the Drive copy is not trusted."""
    bob = PeerStore(email="bob@example.com", encryption_policy=REQUIRED)
    bob.generate_keys()
    store = PeerStore(email="alice@example.com")
    store.set_peers(
        [
            Peer(
                email="bob@example.com",
                public_encryption_bundle=bob.get_public_bundle(),
            )
        ]
    )
    assert not store.has_peer_bundle("bob@example.com")


def test_optional_client_pins_published_bundle_over_drive_copy():
    """A swapped bundle in SYFT_peers.json does not become the key we encrypt to."""
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        add_peers=False,
        do_encryption_policy=REQUIRED,
    )
    ds_manager.add_peer(do_manager.email, sync=False)
    do_manager.load_peers()
    do_manager.approve_peer_request(ds_manager.email)

    attacker = PeerStore(email=do_manager.email, encryption_policy=REQUIRED)
    attacker.generate_keys()
    ds_manager.peer_manager.connection_router.update_peer_state(
        do_manager.email,
        PeerState.REQUESTED_BY_ME.value,
        public_encryption_bundle=attacker.get_public_bundle(),
    )
    ds_manager.load_peers()

    ds_store = ds_manager.peer_manager.peer_store
    real = do_manager.peer_manager.peer_store.my_fingerprint
    assert ds_store.peer_fingerprint(do_manager.email) == real


# =========================================================================
# Receive path
# =========================================================================


@pytest.mark.parametrize(
    "alice_policy, bob_policy",
    [(REQUIRED, OPTIONAL), (OPTIONAL, REQUIRED)],
    ids=["optional receiver", "required receiver"],
)
def test_receiver_rejects_plaintext_when_pair_encrypts(alice_policy, bob_policy):
    """A pair that encrypts never sends plaintext, so plaintext is a downgrade."""
    _, bob = _keyed_pair(alice_policy, bob_policy)
    with pytest.raises(ValueError, match="not encrypted"):
        bob.decrypt_and_verify_if_needed("alice@example.com", b"plain")


def test_receiver_accepts_plaintext_in_plaintext_pair():
    _, bob = _keyed_pair(OPTIONAL, OPTIONAL)
    assert bob.decrypt_and_verify_if_needed("alice@example.com", b"plain") == b"plain"


def test_receiver_accepts_envelope_from_plaintext_peer():
    alice, bob = _keyed_pair(REQUIRED, OPTIONAL)
    envelope = alice.encrypt("bob@example.com", b"secret")
    # Bob now sees Alice as optional, e.g. after she restarted without encryption.
    bob.get_cached_peer("alice@example.com").version = _version_with_policy(OPTIONAL)
    assert bob.decrypt_and_verify_if_needed("alice@example.com", envelope) == b"secret"


def _unpinned_receiver() -> PeerStore:
    _, bob = _keyed_pair(REQUIRED, OPTIONAL)
    bob.get_cached_peer("alice@example.com").public_encryption_bundle = None
    return bob


def _keyless_receiver() -> PeerStore:
    bob = PeerStore(email="bob@example.com")
    bob.add_peer(
        Peer(email="alice@example.com", version=_version_with_policy(REQUIRED))
    )
    return bob


@pytest.mark.parametrize(
    "make_receiver",
    [_unpinned_receiver, _keyless_receiver],
    ids=["no peer bundle", "no own keys"],
)
def test_envelope_before_keys_and_pin_is_not_ready(make_receiver):
    alice, _ = _keyed_pair(REQUIRED, OPTIONAL)
    envelope = alice.encrypt("bob@example.com", b"secret")
    bob = make_receiver()
    with pytest.raises(PeerKeysNotReadyError):
        bob.decrypt_and_verify_if_needed("alice@example.com", envelope)


# =========================================================================
# End to end over the mock drive
# =========================================================================


@pytest.mark.parametrize(
    "ds_policy, do_policy",
    [(OPTIONAL, REQUIRED), (REQUIRED, OPTIONAL), (REQUIRED, REQUIRED)],
)
def test_negotiated_encryption_flow(ds_policy, do_policy):
    """The optional side complies with no manual configuration."""
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        ds_encryption_policy=ds_policy,
        do_encryption_policy=do_policy,
    )
    grant_job_inbox_access(do_manager, ds_manager.email)
    # One load_peers round each lets both sides see the policy and pin bundles.
    ds_manager.load_peers()
    do_manager.load_peers()

    file_path = path_for_job(do_manager.email, ds_manager.email, "my.job")
    ds_manager._send_file_change(file_path, "negotiated content")
    assert _is_envelope(_raw_inbox_message(do_manager, ds_manager.email))

    do_manager.sync()
    assert any("negotiated content" in c for c in _accepted_contents(do_manager))

    ds_manager.sync()
    assert len(_ds_cached_contents(ds_manager)) > 0


def test_client_that_becomes_required_after_peering_publishes_its_bundle():
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection()
    grant_job_inbox_access(do_manager, ds_manager.email)

    # A later login of the DS requires encryption.
    ds_manager.peer_manager.peer_store.encryption_policy = REQUIRED
    ds_manager.peer_manager.write_own_version()
    ds_manager.load_peers()
    do_manager.load_peers()
    ds_manager.load_peers()

    do_store = do_manager.peer_manager.peer_store
    assert do_store.encryption_ready(ds_manager.email)
    file_path = path_for_job(do_manager.email, ds_manager.email, "my.job")
    ds_manager._send_file_change(file_path, "after switch")
    do_manager.sync()
    assert any("after switch" in c for c in _accepted_contents(do_manager))


def test_policy_recorded_by_another_process_is_taken_at_next_sync():
    """syft-bg starts required next to a notebook that started optional."""
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection()
    grant_job_inbox_access(do_manager, ds_manager.email)
    file_path = path_for_job(do_manager.email, ds_manager.email, "my.job")
    ds_manager._send_file_change(file_path, "written while optional")
    do_manager.sync()

    keys_path = do_manager.peer_manager.peer_store._keys_path
    with pytest.warns(UserWarning):
        PeerStore.create(
            email=do_manager.email, encryption_policy=REQUIRED, keys_path=keys_path
        )
    do_manager.sync()

    do_store = do_manager.peer_manager.peer_store
    assert do_store.encryption_policy == REQUIRED
    assert all(_is_envelope(raw) for raw in _raw_event_log(do_manager))
    assert any("written while optional" in c for c in _accepted_contents(do_manager))
    advertised = ds_manager.peer_manager.load_peer_version(do_manager.email)
    assert advertised.encryption_policy == REQUIRED


def _raw_event_log(do_manager) -> list[bytes]:
    connection = (
        do_manager.datasite_owner_syncer.connection_router.connection_for_eventlog()
    )
    return connection.owner_download_all_raw_events_from_syftbox()


def test_owner_that_becomes_required_encrypts_files_at_rest():
    """Plaintext written while optional is encrypted once, then read as usual."""
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection()
    grant_job_inbox_access(do_manager, ds_manager.email)
    file_path = path_for_job(do_manager.email, ds_manager.email, "my.job")
    ds_manager._send_file_change(file_path, "before switch")
    do_manager.sync()
    do_manager.create_checkpoint()
    assert not any(_is_envelope(raw) for raw in _raw_event_log(do_manager))

    do_manager.peer_manager.peer_store.encryption_policy = REQUIRED
    do_manager.peer_manager.peer_store.ensure_keys()
    do_manager.sync()

    assert all(_is_envelope(raw) for raw in _raw_event_log(do_manager))
    assert any("before switch" in c for c in _accepted_contents(do_manager))
    router = do_manager.datasite_owner_syncer.connection_router
    assert router.get_latest_checkpoint() is not None


def test_off_client_works_in_plaintext_with_optional_peer():
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        ds_encryption_policy=OFF,
        do_encryption_policy=OPTIONAL,
    )
    grant_job_inbox_access(do_manager, ds_manager.email)
    assert ds_manager.peer_manager.get_own_version().encryption_policy == OFF

    file_path = path_for_job(do_manager.email, ds_manager.email, "my.job")
    ds_manager._send_file_change(file_path, "plain content")
    assert not _is_envelope(_raw_inbox_message(do_manager, ds_manager.email))
    do_manager.sync()
    assert any("plain content" in c for c in _accepted_contents(do_manager))
    assert not ds_manager.peer_manager.peer_store.has_my_keys()


def test_off_and_required_peers_skip_each_other():
    """Both syncs go on without the other side, and say why."""
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        ds_encryption_policy=OFF,
        do_encryption_policy=REQUIRED,
    )
    grant_job_inbox_access(do_manager, ds_manager.email)
    ds_manager.load_peers()

    file_path = path_for_job(do_manager.email, ds_manager.email, "my.job")
    with pytest.raises(EncryptionPolicyConflictError):
        ds_manager._send_file_change(file_path, "content")
    with pytest.warns(UserWarning, match="skipped"):
        do_manager.sync()
    with pytest.warns(UserWarning, match="skipped"):
        ds_manager.sync()


def test_owner_drops_outbox_message_for_conflicting_peer():
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        ds_encryption_policy=OFF,
        do_encryption_policy=REQUIRED,
    )
    syncer = do_manager.datasite_owner_syncer
    syncer.outbox_queue.put((ds_manager.email, FileChangeEventsMessage(events=[])))
    syncer.process_syftbox_events_queue()
    assert syncer.held_message_count == 0


def test_conflict_found_on_read_makes_the_peer_wait():
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        ds_encryption_policy=OFF,
        do_encryption_policy=OPTIONAL,
    )
    grant_job_inbox_access(do_manager, ds_manager.email)
    file_path = path_for_job(do_manager.email, ds_manager.email, "my.job")
    ds_manager._send_file_change(file_path, "content")

    # The owner now requires encryption, after the message was sent.
    do_manager.peer_manager.peer_store.encryption_policy = REQUIRED
    syncer = do_manager.datasite_owner_syncer
    assert (
        syncer.pull_and_process_next_proposed_filechange(
            ds_manager.email, raise_on_none=False
        )
        is None
    )


def test_owner_waits_on_envelope_from_unpinned_peer():
    """The sync goes on; the message stays in the inbox until the pin exists."""
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        ds_encryption_policy=REQUIRED,
        do_encryption_policy=OPTIONAL,
    )
    grant_job_inbox_access(do_manager, ds_manager.email)
    do_manager.load_peers()
    file_path = path_for_job(do_manager.email, ds_manager.email, "my.job")
    ds_manager._send_file_change(file_path, "pinned later")

    peer = do_manager.peer_manager.peer_store.get_cached_peer(ds_manager.email)
    peer.public_encryption_bundle = None
    do_manager.datasite_owner_syncer.sync([ds_manager.email])
    assert not any("pinned later" in c for c in _accepted_contents(do_manager))

    do_manager.sync()
    assert any("pinned later" in c for c in _accepted_contents(do_manager))


# =========================================================================
# Held sends
# =========================================================================


def test_send_rereads_unknown_peer_policy():
    """The peer can have accepted since the last load; one re-read finds its policy."""
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        ds_encryption_policy=OPTIONAL,
        do_encryption_policy=REQUIRED,
    )
    grant_job_inbox_access(do_manager, ds_manager.email)
    ds_manager.load_peers()
    ds_manager.peer_manager.get_cached_peer(do_manager.email).version = None

    file_path = path_for_job(do_manager.email, ds_manager.email, "my.job")
    ds_manager._send_file_change(file_path, "content")
    assert _is_envelope(_raw_inbox_message(do_manager, ds_manager.email))


def test_send_before_peer_accepts_is_held_until_sync(capsys):
    """Before the owner accepts, its policy is unknown, so the send waits."""
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        add_peers=False,
        do_encryption_policy=REQUIRED,
    )
    ds_manager.add_peer(do_manager.email, sync=False)

    file_path = path_for_job(do_manager.email, ds_manager.email, "my.job")
    ds_manager._send_file_change(file_path, "held content")
    watcher = ds_manager.datasite_watcher_syncer
    assert watcher.pending_message_count == 1
    assert "Held a message" in capsys.readouterr().out
    # The held message is on disk, so a restart keeps it.
    assert (ds_manager.syftbox_folder / ".cache" / "held_proposals.json").exists()

    do_manager.load_peers()
    do_manager.approve_peer_request(ds_manager.email)
    grant_job_inbox_access(do_manager, ds_manager.email)
    ds_manager.sync()
    assert watcher.pending_message_count == 0
    assert _is_envelope(_raw_inbox_message(do_manager, ds_manager.email))

    do_manager.sync()
    assert any("held content" in c for c in _accepted_contents(do_manager))


def test_send_after_accept_before_peer_load_is_held():
    """The owner accepted, but the scientist has no keys or pin yet: hold, then send."""
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        add_peers=False,
        do_encryption_policy=REQUIRED,
    )
    ds_manager.add_peer(do_manager.email, sync=False)
    do_manager.load_peers()
    do_manager.approve_peer_request(ds_manager.email)
    grant_job_inbox_access(do_manager, ds_manager.email)

    file_path = path_for_job(do_manager.email, ds_manager.email, "my.job")
    ds_manager._send_file_change(file_path, "after accept")
    assert ds_manager.datasite_watcher_syncer.pending_message_count == 1

    ds_manager.sync()
    do_manager.sync()
    assert any("after accept" in c for c in _accepted_contents(do_manager))


def test_held_change_does_not_update_hash_cache():
    """The cache records what the owner accepted, so a restart re-proposes correctly."""
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        add_peers=False,
    )
    ds_manager.add_peer(do_manager.email, sync=False)
    file_path = path_for_job(do_manager.email, ds_manager.email, "my.job")
    ds_manager._send_file_change(file_path, "held content")

    cache = ds_manager.datasite_watcher_syncer.datasite_watcher_cache
    assert Path(file_path) not in cache.file_hashes


def test_two_held_changes_to_one_file_apply_in_order():
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        add_peers=False,
    )
    ds_manager.add_peer(do_manager.email, sync=False)
    file_path = path_for_job(do_manager.email, ds_manager.email, "my.job")
    ds_manager._send_file_change(file_path, "first")
    ds_manager._send_file_change(file_path, "second")
    assert ds_manager.datasite_watcher_syncer.pending_message_count == 2

    do_manager.load_peers()
    do_manager.approve_peer_request(ds_manager.email)
    grant_job_inbox_access(do_manager, ds_manager.email)
    ds_manager.sync()
    do_manager.sync()
    assert any("second" in c for c in _accepted_contents(do_manager))


def test_send_to_non_peer_raises():
    """A recipient that is not a peer never gets a policy, so holding would be forever."""
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        add_peers=False,
    )
    file_path = path_for_job(do_manager.email, ds_manager.email, "my.job")
    with pytest.raises(PeerEncryptionPolicyUnknownError):
        ds_manager._send_file_change(file_path, "content")
    assert ds_manager.datasite_watcher_syncer.pending_message_count == 0


def test_changes_for_two_datasites_go_to_each(monkeypatch):
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection()
    sent = []
    monkeypatch.setattr(
        DatasiteWatcherSyncer,
        "_send_or_hold",
        lambda self, recipient, message: sent.append(
            (recipient, {fc.datasite_email for fc in message.proposed_file_changes})
        )
        or True,
    )
    watcher = ds_manager.datasite_watcher_syncer
    watcher.on_file_change(f"{do_manager.email}/a.txt", "a", process_now=False)
    watcher.on_file_change("other@example.com/b.txt", "b", process_now=False)
    watcher.process_file_changes_queue()
    assert sorted(sent) == sorted(
        [
            (do_manager.email, {do_manager.email}),
            ("other@example.com", {"other@example.com"}),
        ]
    )


def test_conflict_with_one_datasite_still_sends_to_the_other(monkeypatch):
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection()
    sent = []

    def send_or_hold(self, recipient, message):
        if recipient == "conflict@example.com":
            raise EncryptionPolicyConflictError(recipient)
        sent.append(recipient)
        return True

    monkeypatch.setattr(DatasiteWatcherSyncer, "_send_or_hold", send_or_hold)
    watcher = ds_manager.datasite_watcher_syncer
    watcher.on_file_change("conflict@example.com/a.txt", "a", process_now=False)
    watcher.on_file_change(f"{do_manager.email}/b.txt", "b", process_now=False)
    with pytest.raises(EncryptionPolicyConflictError):
        watcher.process_file_changes_queue()
    assert sent == [do_manager.email]


def test_owner_drops_outbox_message_for_former_peer():
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection()
    syncer = do_manager.datasite_owner_syncer
    syncer.outbox_queue.put(("former@example.com", FileChangeEventsMessage(events=[])))
    syncer.process_syftbox_events_queue()
    assert syncer.held_message_count == 0


def test_policy_is_reread_once_per_peer_load(monkeypatch):
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection()
    syncer = do_manager.datasite_owner_syncer
    do_manager.peer_manager.get_cached_peer(ds_manager.email).version = None
    reads = []
    monkeypatch.setattr(
        ConnectionRouter,
        "read_peer_version_file",
        lambda self, email: reads.append(email),
    )
    for _ in range(3):
        syncer.outbox_queue.put((ds_manager.email, FileChangeEventsMessage(events=[])))
    syncer.process_syftbox_events_queue()
    syncer.process_syftbox_events_queue()
    assert reads == [ds_manager.email]


def test_compaction_skips_peer_with_unknown_policy(monkeypatch):
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection()
    grant_job_inbox_access(do_manager, ds_manager.email)
    file_path = path_for_job(do_manager.email, ds_manager.email, "my.job")
    for content in ("one", "two"):
        ds_manager._send_file_change(file_path, content)
        do_manager.sync()

    do_manager.peer_manager.get_cached_peer(ds_manager.email).version = None
    monkeypatch.setattr(
        ConnectionRouter, "read_peer_version_file", lambda self, email: None
    )
    syncer = do_manager.datasite_owner_syncer
    assert syncer.compact_outbox_if_needed(ds_manager.email, min_messages=1) == 0


def _unknown_policy(ds_manager, do_manager, monkeypatch):
    """Make the DS policy unknown to the DO; return a function that undoes it."""
    peer = do_manager.peer_manager.get_cached_peer(ds_manager.email)
    version = peer.version
    peer.version = None
    monkeypatch.setattr(
        ConnectionRouter, "read_peer_version_file", lambda self, email: None
    )

    def restore():
        monkeypatch.undo()
        peer.version = version

    return restore


def _unpinned_bundle(ds_manager, do_manager, monkeypatch):
    """Drop the DO pin of the DS bundle; return a function that undoes it."""
    peer = do_manager.peer_manager.get_cached_peer(ds_manager.email)
    bundle = peer.public_encryption_bundle
    peer.public_encryption_bundle = None

    def restore():
        peer.public_encryption_bundle = bundle

    return restore


@pytest.mark.parametrize(
    "make_pending", [_unknown_policy, _unpinned_bundle], ids=["policy", "pin"]
)
def test_owner_holds_outbox_message_until_encryption_is_set_up(
    make_pending, monkeypatch
):
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        ds_encryption_policy=REQUIRED,
    )
    syncer = do_manager.datasite_owner_syncer
    restore = make_pending(ds_manager, do_manager, monkeypatch)

    syncer.outbox_queue.put((ds_manager.email, FileChangeEventsMessage(events=[])))
    syncer.process_syftbox_events_queue()
    assert syncer.held_message_count == 1
    # The held message is on disk, so a restart keeps it.
    assert (do_manager.syftbox_folder / ".cache" / "held_outbox.json").exists()

    restore()
    syncer.process_syftbox_events_queue()
    assert syncer.held_message_count == 0


# =========================================================================
# Rejected messages
# =========================================================================


def test_rejected_inbox_message_is_resent_in_order():
    """A newer encrypted change waits until the rejected one is resent encrypted."""
    ds_manager, do_manager = _pair_with_stale_sender_view()
    file_path = path_for_job(do_manager.email, ds_manager.email, "my.job")
    ds_manager._send_file_change(file_path, "first")
    do_manager.sync()

    # The sender learns the policy and sends a newer change before it resends.
    ds_manager.load_peers()
    ds_manager._send_file_change(file_path, "second")
    do_manager.sync()
    assert not any("second" in c for c in _accepted_contents(do_manager))

    ds_manager.sync()
    assert _is_envelope(_raw_inbox_message(do_manager, ds_manager.email))
    do_manager.sync()
    contents = _accepted_contents(do_manager)
    assert any("first" in c for c in contents)
    assert any("second" in c for c in contents)


def test_rejected_outbox_message_is_resent_in_order(monkeypatch):
    ds_manager, do_manager = _required_ds_pair()
    _owner_sends_with_stale_view(ds_manager, do_manager, "first")
    ds_manager.sync()
    assert not any("first" in c for c in _ds_cached_contents(ds_manager))

    # The owner refreshes its view and answers a newer proposal, but has not
    # resent the rejected message yet.
    do_manager.load_peers()
    with monkeypatch.context() as m:
        m.setattr(
            ConnectionRouter,
            "owner_resend_rejected_outbox_messages",
            lambda self, recipient: 0,
        )
        file_path = path_for_job(do_manager.email, ds_manager.email, "other.job")
        ds_manager._send_file_change(file_path, "second")
        do_manager.datasite_owner_syncer.sync([ds_manager.email])
    ds_manager.sync()
    assert not any("second" in c for c in _ds_cached_contents(ds_manager))

    do_manager.sync()
    ds_manager.sync()
    contents = _ds_cached_contents(ds_manager)
    assert any("first" in c for c in contents)
    assert any("second" in c for c in contents)


def test_all_plaintext_inbox_messages_are_rejected_in_one_pass():
    ds_manager, do_manager = _pair_with_stale_sender_view()
    for name in ("a.job", "b.job", "c.job"):
        file_path = path_for_job(do_manager.email, ds_manager.email, name)
        ds_manager._send_file_change(file_path, f"content {name}")
    do_manager.sync()

    ds_manager.sync()
    do_manager.sync()
    contents = _accepted_contents(do_manager)
    for name in ("a.job", "b.job", "c.job"):
        assert any(f"content {name}" in c for c in contents)


def test_message_rejected_twice_keeps_one_suffix():
    """A notebook and syft-bg can both reject one message."""
    ds_manager, do_manager = _pair_with_stale_sender_view()
    file_path = path_for_job(do_manager.email, ds_manager.email, "my.job")
    ds_manager._send_file_change(file_path, "rejected twice")
    do_manager.sync()

    ds_conn = ds_manager.datasite_watcher_syncer.connection_router.connection_for_send_message()
    [(file_id, _, _)] = ds_conn.watcher_get_rejected_inbox_messages(do_manager.email)
    do_conn = do_manager.datasite_owner_syncer.connection_router.connection_for_receive_message()
    do_conn.mark_message_rejected(file_id)

    ds_manager.sync()
    do_manager.sync()
    assert any("rejected twice" in c for c in _accepted_contents(do_manager))


def test_optional_pair_does_not_look_for_rejected_messages(monkeypatch):
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection()
    calls = []
    original = GDriveConnection.watcher_get_rejected_inbox_messages

    def spy(self, recipient):
        calls.append(recipient)
        return original(self, recipient)

    monkeypatch.setattr(GDriveConnection, "watcher_get_rejected_inbox_messages", spy)
    ds_manager.sync()
    assert calls == []


def test_resend_waits_while_peer_bundle_is_not_pinned():
    ds_manager, do_manager = _pair_with_stale_sender_view()
    file_path = path_for_job(do_manager.email, ds_manager.email, "my.job")
    ds_manager._send_file_change(file_path, "plain content")
    do_manager.sync()

    peer = ds_manager.peer_manager.get_cached_peer(do_manager.email)
    peer.version = _version_with_policy(REQUIRED)
    peer.public_encryption_bundle = None
    router = ds_manager.datasite_watcher_syncer.connection_router
    assert router.watcher_resend_rejected_inbox_messages(do_manager.email) == 0
