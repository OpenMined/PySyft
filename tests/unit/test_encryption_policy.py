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
from syft.sync.encryption_policy import (
    EncryptionPolicy,
    EncryptionPolicyConflictError,
    PeerOutdatedError,
    PeerEncryptionPolicyUnknownError,
    PeerKeysNotReadyError,
)
from syft.sync.events.file_change_event import (
    FileChangeEventsMessage,
)
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


def _forget(store: PeerStore, email: str) -> None:
    """Make ``store`` forget what it knew of ``email``'s policy, as a new client."""
    store._last_known_policies.pop(email, None)


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
    # A client that never saw the owner as required.
    _forget(ds_manager.peer_manager.peer_store, do_manager.email)
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
    _forget(do_manager.peer_manager.peer_store, ds_manager.email)
    file_path = path_for_job(do_manager.email, ds_manager.email, "my.job")
    ds_manager._send_file_change(file_path, content)
    do_manager.datasite_owner_syncer.sync([ds_manager.email])


# =========================================================================
# Policy model and advertisement
# =========================================================================


def test_policy_defaults_to_optional():
    assert PeerStore(email="alice@example.com").encryption_policy == OPTIONAL


@pytest.mark.parametrize(
    "value, expected",
    [(True, REQUIRED), (False, OPTIONAL), ("off", OFF), (OFF, OFF)],
)
def test_encryption_setting_coerced_to_policy(value, expected):
    assert EncryptionPolicy.coerce(value) == expected


def test_version_info_without_policy_field_has_no_policy():
    """A version file from a client that predates the field says nothing."""
    data = VersionInfo.current().model_dump(mode="json")
    data.pop("encryption_policy")
    assert VersionInfo.model_validate(data).encryption_policy is None


def _advertise_as_outdated_client(manager) -> None:
    """Write the version file of ``manager`` without the policy field."""
    outdated = manager.peer_manager.get_own_version().model_copy()
    outdated.encryption_policy = None
    manager.peer_manager.connection_router.write_version_file(outdated)


def test_outdated_peer_skipped_with_upgrade_warning():
    """A peer without a policy in its version file runs an outdated syft."""
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection()
    grant_job_inbox_access(do_manager, ds_manager.email)
    _advertise_as_outdated_client(do_manager)
    ds_manager.load_peers()

    file_path = path_for_job(do_manager.email, ds_manager.email, "my.job")
    with pytest.raises(PeerOutdatedError, match="upgrade"):
        ds_manager._send_file_change(file_path, "content")
    with pytest.warns(UserWarning, match="upgrade"):
        ds_manager.sync()


def test_own_version_advertises_policy():
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        ds_encryption_policy=OPTIONAL,
        do_encryption_policy=REQUIRED,
    )
    assert do_manager.peer_manager.get_own_version().encryption_policy == REQUIRED
    assert ds_manager.peer_manager.get_own_version().encryption_policy == OPTIONAL
    advertised = ds_manager.peer_manager.get_peer_version(do_manager.email)
    assert advertised.encryption_policy == REQUIRED


def test_missing_version_file_created_with_own_policy():
    a_conn, b_conn = mock_drive_service.pair_with_mock_service(
        "a@example.com", "b@example.com"
    )
    a_conn.share_version_file_with_peer("b@example.com", _version_with_policy(REQUIRED))
    assert b_conn.read_peer_version_file("a@example.com").encryption_policy == REQUIRED


def _write_garbage_own_bundle(ds_manager, do_manager, monkeypatch) -> None:
    ds_manager.peer_manager.connection_router.write_encryption_bundle(
        do_manager.email, "not json"
    )


def _make_own_bundle_undecodable(ds_manager, do_manager, monkeypatch) -> None:
    def undecodable(self, peer_email):
        raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")

    monkeypatch.setattr(ConnectionRouter, "read_own_encryption_bundle", undecodable)


@pytest.mark.parametrize(
    "damage",
    [_write_garbage_own_bundle, _make_own_bundle_undecodable],
    ids=["not json", "not utf-8"],
)
def test_unreadable_own_bundle_is_written_again(damage, monkeypatch):
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        do_encryption_policy=REQUIRED,
    )
    damage(ds_manager, do_manager, monkeypatch)
    ds_manager.peer_manager._bundle_published_for.clear()  # as a new process
    ds_manager.load_peers()
    assert do_manager.email in ds_manager.peer_manager._bundle_published_for


def test_malformed_peer_bundle_does_not_stop_peer_load():
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        add_peers=False,
        do_encryption_policy=REQUIRED,
    )
    ds_manager.add_peer(do_manager.email, sync=False)
    do_manager.load_peers()
    do_manager.approve_peer_request(ds_manager.email)
    do_manager.peer_manager.connection_router.write_encryption_bundle(
        ds_manager.email, "not json"
    )
    with pytest.warns(UserWarning, match="Refusing"):
        ds_manager.load_peers()
    assert not ds_manager.peer_manager.peer_store.has_peer_bundle(do_manager.email)


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
def test_peer_encryption_resolved_from_both_policies(mine, theirs, encrypted):
    store = _store_with_peer(mine, theirs)
    assert store.peer_uses_encryption("bob@example.com") is encrypted


@pytest.mark.parametrize(
    "mine, theirs", [(OFF, REQUIRED), (REQUIRED, OFF)], ids=["off", "required"]
)
def test_off_and_required_conflict(mine, theirs):
    """Neither side gives way, so the pair cannot exchange messages."""
    alice, bob = _keyed_pair(mine, theirs)
    peer = alice.get_cached_peer("bob@example.com")
    assert isinstance(alice.conflict_error(peer), EncryptionPolicyConflictError)
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
def test_encrypting_pair_without_keys_or_pin_not_ready(mine, theirs, with_keys):
    store = _store_with_peer(mine, theirs)
    if with_keys:
        store.generate_keys()
    with pytest.raises(PeerKeysNotReadyError):
        store.encrypt_if_needed("bob@example.com", b"data")


# =========================================================================
# Keys and pins
# =========================================================================


@pytest.mark.parametrize("policy", [OPTIONAL, OFF])
def test_store_not_requiring_encryption_creates_no_key_file(tmp_path, policy):
    keys_path = tmp_path / "crypto_keys.json"
    store = PeerStore.create(
        email="alice@example.com", encryption_policy=policy, keys_path=keys_path
    )
    assert not store.has_my_keys()
    assert not keys_path.exists()


@pytest.mark.parametrize("policy", [OPTIONAL, OFF])
def test_sync_refuses_key_file_made_required_after_start(policy):
    """The policy is read at startup; a running process must not go on without it."""
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        do_encryption_policy=policy,
    )
    keys_path = do_manager.peer_manager.peer_store._keys_path
    with pytest.warns(UserWarning):
        PeerStore.create(
            email=do_manager.email, encryption_policy=REQUIRED, keys_path=keys_path
        )
    with pytest.raises(ValueError, match="Restart"):
        do_manager.sync()


def test_off_refused_when_key_file_requires_encryption(tmp_path):
    """Files at rest of such a datasite are encrypted; off cannot read them."""
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


def _create(keys_path, policy=OPTIONAL) -> PeerStore:
    return PeerStore.create(
        email="alice@example.com", encryption_policy=policy, keys_path=keys_path
    )


def _make_optional_keys(keys_path) -> None:
    _create(keys_path).ensure_keys()


def _first_required_login(keys_path) -> None:
    with pytest.warns(UserWarning):
        _create(keys_path, REQUIRED)


@pytest.mark.parametrize(
    "before", [lambda keys_path: None, _make_optional_keys], ids=["first", "switch"]
)
def test_datasite_starting_to_require_warns_about_key_file(tmp_path, before):
    keys_path = tmp_path / "crypto_keys.json"
    before(keys_path)
    with pytest.warns(UserWarning, match="keep a copy"):
        _create(keys_path, REQUIRED)


@pytest.mark.parametrize(
    "before, action",
    [
        (_first_required_login, lambda keys_path: _create(keys_path, REQUIRED)),
        (_first_required_login, lambda keys_path: _create(keys_path, OPTIONAL)),
        (lambda keys_path: None, _make_optional_keys),
    ],
    ids=["later required login", "later optional login", "optional keys"],
)
def test_no_key_file_warning_without_switch_to_required(tmp_path, before, action):
    keys_path = tmp_path / "crypto_keys.json"
    before(keys_path)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        action(keys_path)


def test_keys_from_other_process_keep_pin_held_in_memory(tmp_path):
    """A pin set before there were keys is saved once the key file exists."""
    keys_path = tmp_path / "crypto_keys.json"
    bob = PeerStore(email="bob@example.com", encryption_policy=REQUIRED)
    bob.generate_keys()
    alice = PeerStore.create(email="alice@example.com", keys_path=keys_path)
    alice.add_peer(
        Peer(email="bob@example.com", version=_version_with_policy(REQUIRED))
    )
    alice.set_peer_bundle("bob@example.com", bob.get_public_bundle())
    # Another process on the datasite makes the key file.
    PeerStore.create(email="alice@example.com", keys_path=keys_path).ensure_keys()

    thread = threading.Thread(target=alice.ensure_keys, daemon=True)
    thread.start()
    thread.join(timeout=10)
    assert not thread.is_alive(), "ensure_keys did not finish"
    assert PeerStore.load_keys(keys_path).has_peer_bundle("bob@example.com")


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


def test_plaintext_from_peer_with_unknown_policy_waits():
    """The receive path waits like the send path while the policy is unknown."""
    _, bob = _keyed_pair(OPTIONAL, OPTIONAL)
    bob.get_cached_peer("alice@example.com").version = None
    with pytest.raises(PeerEncryptionPolicyUnknownError):
        bob.decrypt_and_verify_if_needed("alice@example.com", b"plain")


def test_last_known_policy_kept_when_version_vanishes():
    """E.g. after the owner's delete_syftbox(): its plaintext deletes still apply."""
    _, bob = _keyed_pair(OPTIONAL, OPTIONAL)
    assert bob.peer_policy("alice@example.com") == OPTIONAL
    bob.get_cached_peer("alice@example.com").version = None
    assert bob.decrypt_and_verify_if_needed("alice@example.com", b"plain") == b"plain"


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


def test_client_required_after_peering_publishes_bundle():
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


def _raw_event_log(do_manager) -> list[bytes]:
    connection = (
        do_manager.datasite_owner_syncer.connection_router.connection_for_eventlog()
    )
    return connection.owner_download_all_raw_events_from_syftbox()


def test_owner_switching_to_required_encrypts_new_files_at_rest():
    """Files written while optional stay plaintext, and are read as usual."""
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection()
    grant_job_inbox_access(do_manager, ds_manager.email)
    file_path = path_for_job(do_manager.email, ds_manager.email, "my.job")
    ds_manager._send_file_change(file_path, "before switch")
    do_manager.sync()
    do_manager.create_checkpoint()
    before = len(_raw_event_log(do_manager))

    # A later login of the DO requires encryption.
    do_manager.peer_manager.peer_store.encryption_policy = REQUIRED
    do_manager.peer_manager.write_own_version()
    do_manager.load_peers()
    ds_manager.load_peers()
    do_manager.load_peers()
    ds_manager._send_file_change(file_path, "after switch")
    do_manager.sync()

    envelopes = [_is_envelope(raw) for raw in _raw_event_log(do_manager)]
    assert envelopes.count(False) == before
    assert envelopes.count(True) == len(envelopes) - before > 0
    contents = _accepted_contents(do_manager)
    assert any("before switch" in c for c in contents)
    assert any("after switch" in c for c in contents)
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


@pytest.mark.parametrize(
    "ds_policy, recipient",
    [(OPTIONAL, "former@example.com"), (OFF, None)],
    ids=["former peer", "conflicting peer"],
)
def test_owner_drops_outbox_message_it_cannot_send(ds_policy, recipient):
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        ds_encryption_policy=ds_policy,
        do_encryption_policy=REQUIRED,
    )
    syncer = do_manager.datasite_owner_syncer
    message = FileChangeEventsMessage(events=[])
    syncer.outbox_queue.put((recipient or ds_manager.email, message))
    with pytest.warns(UserWarning, match="Dropped an outbox message"):
        syncer.process_syftbox_events_queue()
    assert syncer.outbox_queue.empty()


def test_conflict_found_on_read_makes_peer_wait():
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
    with pytest.warns(UserWarning, match=f"Messages from {ds_manager.email} wait"):
        message = syncer.pull_and_process_next_proposed_filechange(
            ds_manager.email, raise_on_none=False
        )
    assert message is None


def test_current_published_bundle_not_written_again(monkeypatch):
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        do_encryption_policy=REQUIRED,
    )
    ds_manager.load_peers()
    writes = []
    monkeypatch.setattr(
        ConnectionRouter,
        "write_encryption_bundle",
        lambda self, peer_email, bundle_json: writes.append(peer_email),
    )
    ds_manager.peer_manager._bundle_published_for.clear()  # as a new process
    ds_manager.load_peers()
    assert writes == []


def test_drive_error_while_publishing_bundle_does_not_stop_sync(monkeypatch):
    from googleapiclient.errors import HttpError

    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        do_encryption_policy=REQUIRED,
    )

    def fail(self, peer_email):
        raise HttpError(
            resp=type("R", (), {"status": 503, "reason": ""})(), content=b""
        )

    monkeypatch.setattr(ConnectionRouter, "read_own_encryption_bundle", fail)
    ds_manager.peer_manager._bundle_published_for.clear()  # as a new process
    with pytest.warns(UserWarning, match="publish"):
        ds_manager.load_peers()
    assert do_manager.email not in ds_manager.peer_manager._bundle_published_for


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
    with pytest.warns(UserWarning, match="wait"):
        do_manager.datasite_owner_syncer.sync([ds_manager.email])
    assert not any("pinned later" in c for c in _accepted_contents(do_manager))

    do_manager.sync()
    assert any("pinned later" in c for c in _accepted_contents(do_manager))


# =========================================================================
# Sends while encryption is pending
# =========================================================================


def test_send_reread_updates_peer_manager():
    """The re-read also refreshes what PeerManager derives from the version."""
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        ds_encryption_policy=OPTIONAL,
        do_encryption_policy=REQUIRED,
    )
    grant_job_inbox_access(do_manager, ds_manager.email)
    ds_manager.load_peers()
    peer_manager = ds_manager.peer_manager
    peer_manager.get_cached_peer(do_manager.email).version = None
    _forget(peer_manager.peer_store, do_manager.email)
    peer_manager._loaded_peer_versions[do_manager.email] = None

    file_path = path_for_job(do_manager.email, ds_manager.email, "my.job")
    ds_manager._send_file_change(file_path, "content")
    assert peer_manager._loaded_peer_versions[do_manager.email] is not None
    assert _is_envelope(_raw_inbox_message(do_manager, ds_manager.email))


def _queued_changes(manager) -> int:
    return manager.datasite_watcher_syncer.queue.qsize()


def test_send_before_peer_accepts_waits_until_sync():
    """Before the owner accepts, its policy is unknown, so the change waits."""
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        add_peers=False,
        do_encryption_policy=REQUIRED,
    )
    ds_manager.add_peer(do_manager.email, sync=False)

    file_path = path_for_job(do_manager.email, ds_manager.email, "my.job")
    with pytest.warns(UserWarning, match="lost if this process exits"):
        ds_manager._send_file_change(file_path, "early content")
    assert _queued_changes(ds_manager) == 1
    cache = ds_manager.datasite_watcher_syncer.datasite_watcher_cache
    assert Path(file_path) not in cache.file_hashes

    do_manager.load_peers()
    do_manager.approve_peer_request(ds_manager.email)
    grant_job_inbox_access(do_manager, ds_manager.email)
    ds_manager.sync()
    assert _queued_changes(ds_manager) == 0
    assert _is_envelope(_raw_inbox_message(do_manager, ds_manager.email))
    do_manager.sync()
    assert any("early content" in c for c in _accepted_contents(do_manager))


def test_send_after_accept_before_peer_load_waits_until_sync():
    """The owner accepted, but the scientist has no keys or pin yet."""
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
    assert _queued_changes(ds_manager) == 1

    ds_manager.sync()
    do_manager.sync()
    assert any("after accept" in c for c in _accepted_contents(do_manager))


def test_send_to_non_peer_raises():
    """A recipient that is not a peer never gets a policy."""
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        add_peers=False,
    )
    file_path = path_for_job(do_manager.email, ds_manager.email, "my.job")
    with pytest.raises(PeerEncryptionPolicyUnknownError):
        ds_manager._send_file_change(file_path, "content")


def test_changes_for_two_datasites_go_to_each(monkeypatch):
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection()
    sent = []
    monkeypatch.setattr(
        DatasiteWatcherSyncer,
        "_send_message",
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


@pytest.mark.parametrize(
    "error",
    [
        EncryptionPolicyConflictError,
        PeerEncryptionPolicyUnknownError,
        PeerKeysNotReadyError,
    ],
    ids=["conflict", "not a peer", "keys not ready"],
)
def test_failure_for_one_recipient_still_sends_to_other(monkeypatch, error):
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection()
    sent = []

    def send_message(self, recipient, message):
        if recipient == "other@example.com":
            raise error(recipient)
        sent.append(recipient)
        return True

    monkeypatch.setattr(DatasiteWatcherSyncer, "_send_message", send_message)
    watcher = ds_manager.datasite_watcher_syncer
    watcher.on_file_change("other@example.com/a.txt", "a", process_now=False)
    watcher.on_file_change(f"{do_manager.email}/b.txt", "b", process_now=False)
    with pytest.raises(error):
        watcher.process_file_changes_queue()
    assert sent == [do_manager.email]


def test_policy_reread_once_per_peer_load(monkeypatch):
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection()
    syncer = do_manager.datasite_owner_syncer
    do_manager.peer_manager.get_cached_peer(ds_manager.email).version = None
    _forget(do_manager.peer_manager.peer_store, ds_manager.email)
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
    _forget(do_manager.peer_manager.peer_store, ds_manager.email)
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
    _forget(do_manager.peer_manager.peer_store, ds_manager.email)
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
def test_owner_sends_outbox_message_once_encryption_set_up(make_pending, monkeypatch):
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        ds_encryption_policy=REQUIRED,
    )
    syncer = do_manager.datasite_owner_syncer
    restore = make_pending(ds_manager, do_manager, monkeypatch)
    written = []
    original = ConnectionRouter.owner_write_event_messages_to_outbox

    def write(self, recipient, message):
        original(self, recipient, message)
        written.append(recipient)

    monkeypatch.setattr(ConnectionRouter, "owner_write_event_messages_to_outbox", write)
    syncer.outbox_queue.put((ds_manager.email, FileChangeEventsMessage(events=[])))
    with pytest.warns(UserWarning, match="lost if this process exits"):
        syncer.process_syftbox_events_queue()
    assert written == []

    restore()
    monkeypatch.setattr(ConnectionRouter, "owner_write_event_messages_to_outbox", write)
    syncer.process_syftbox_events_queue()
    assert written == [ds_manager.email]


# =========================================================================
# Plaintext refused
# =========================================================================


def test_plaintext_inbox_message_dropped_and_later_ones_read():
    ds_manager, do_manager = _pair_with_stale_sender_view()
    file_path = path_for_job(do_manager.email, ds_manager.email, "my.job")
    ds_manager._send_file_change(file_path, "plain content")

    ds_manager.load_peers()
    # The owner never applies the dropped change, so this one is to another file.
    other_path = path_for_job(do_manager.email, ds_manager.email, "other.job")
    ds_manager._send_file_change(other_path, "encrypted content")
    with pytest.warns(UserWarning, match="not encrypted"):
        do_manager.sync()
    contents = _accepted_contents(do_manager)
    assert not any("plain content" in c for c in contents)
    assert any("encrypted content" in c for c in contents)


def test_plaintext_outbox_message_skipped_and_later_ones_read():
    ds_manager, do_manager = _required_ds_pair()
    _owner_sends_with_stale_view(ds_manager, do_manager, "plain content")

    do_manager.load_peers()
    file_path = path_for_job(do_manager.email, ds_manager.email, "other.job")
    ds_manager._send_file_change(file_path, "encrypted content")
    do_manager.datasite_owner_syncer.sync([ds_manager.email])
    with pytest.warns(UserWarning, match="not encrypted"):
        ds_manager.sync()
    contents = _ds_cached_contents(ds_manager)
    assert not any("plain content" in c for c in contents)
    assert any("encrypted content" in c for c in contents)
