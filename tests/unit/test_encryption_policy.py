"""Per-peer encryption negotiated from each side's advertised policy.

A client is ``optional`` (encrypt only when the peer requires it) or
``required``. A pair encrypts when either side requires it.
"""

import json

import pytest
from pydantic import ValidationError

from syft.sync.connections.connection_router import ConnectionRouter
from syft.sync.connections.drive import mock_drive_service
from syft.sync.encryption_policy import (
    EncryptionPolicy,
    MessageFormatError,
)
from syft.sync.events.file_change_event import (
    FileChangeEventsMessage,
)
from syft.sync.peers.exceptions import SyftPeerNotReadyError
from syft.sync.peers.peer import Peer
from syft.sync.peers.peer_store import PeerStore, datasite_crypto_keys_path
from syft.sync.syftbox_manager import SyftboxManager
from syft.sync.version.version_info import VersionInfo
from tests.unit.test_persisted_cache import (
    _assert_ds_receives_after_notebook_sync,
    _pair_with_unsent_do_file,
)
from tests.unit.test_sync_manager import path_for_job
from tests.unit.utils import grant_job_inbox_access

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


def _owner_answers_in_plaintext(ds_manager, do_manager, content, monkeypatch):
    """The owner handles a proposal and answers in plaintext, as a broken client would."""
    file_path = path_for_job(do_manager.email, ds_manager.email, "my.job")
    ds_manager._send_file_change(file_path, content)
    with monkeypatch.context() as m:
        m.setattr(PeerStore, "encrypt_if_needed", lambda self, email, data: data)
        do_manager.datasite_owner_syncer.sync([ds_manager.email])


# =========================================================================
# Policy model and advertisement
# =========================================================================


def test_policy_defaults_to_optional():
    assert PeerStore(email="alice@example.com").encryption_policy == OPTIONAL


def test_policy_cannot_change_after_login():
    store = PeerStore(email="alice@example.com", encryption_policy=OPTIONAL)
    with pytest.raises(ValidationError, match="frozen"):
        store.encryption_policy = REQUIRED


@pytest.mark.parametrize("value, expected", [(True, REQUIRED), (False, OPTIONAL)])
def test_encryption_setting_maps_to_policy(value, expected):
    assert EncryptionPolicy.from_bool(value) == expected


@pytest.mark.parametrize("value", [OPTIONAL, "optional", 1, None])
def test_encryption_setting_must_be_bool(value):
    with pytest.raises(TypeError, match="True or False"):
        EncryptionPolicy.from_bool(value)


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


def test_malformed_peer_bundle_raises_on_peer_load():
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
    with pytest.raises(ValueError):
        ds_manager.load_peers()


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
    ],
)
def test_peer_encryption_resolved_from_both_policies(mine, theirs, encrypted):
    store = _store_with_peer(mine, theirs)
    assert store.peer_uses_encryption("bob@example.com") is encrypted


def test_resolution_follows_changed_peer_policy():
    """An enclave that reboots with another setting is picked up on the next load."""
    store = _store_with_peer(OPTIONAL, OPTIONAL)
    assert not store.peer_uses_encryption("bob@example.com")

    store.get_cached_peer("bob@example.com").version = _version_with_policy(REQUIRED)
    assert store.peer_uses_encryption("bob@example.com")

    store.get_cached_peer("bob@example.com").version = _version_with_policy(OPTIONAL)
    assert not store.peer_uses_encryption("bob@example.com")


@pytest.mark.parametrize("policy", [OPTIONAL, REQUIRED])
def test_sender_refuses_peer_with_unknown_policy(policy):
    store = PeerStore(email="alice@example.com", encryption_policy=policy)
    store.add_peer(Peer(email="bob@example.com"))
    with pytest.raises(SyftPeerNotReadyError, match="not known"):
        store.encrypt_if_needed("bob@example.com", b"data")


def test_peer_version_without_policy_is_not_known():
    store = _store_with_peer(OPTIONAL, None)
    with pytest.raises(SyftPeerNotReadyError, match="not known"):
        store.encrypt_if_needed("bob@example.com", b"data")


def test_encrypting_pair_without_peer_bundle_not_ready():
    store = _store_with_peer(REQUIRED, OPTIONAL)
    store.generate_keys()
    with pytest.raises(SyftPeerNotReadyError):
        store.encrypt_if_needed("bob@example.com", b"data")


# =========================================================================
# Keys and bundles
# =========================================================================


@pytest.mark.parametrize("policy", [OPTIONAL, REQUIRED])
def test_every_policy_creates_key_file(tmp_path, policy):
    keys_path = tmp_path / "crypto_keys.json"
    store = PeerStore.create(
        email="alice@example.com", encryption_policy=policy, keys_path=keys_path
    )
    reloaded = PeerStore.create(email="alice@example.com", keys_path=keys_path)
    assert reloaded.get_public_bundle() == store.get_public_bundle()


def test_delete_syftbox_keeps_keys_on_disk():
    """The client keeps its keys after the wipe, so the next login must find them."""
    _, do_manager = SyftboxManager.pair_with_mock_drive_service_connection()
    store = do_manager.peer_manager.peer_store
    do_manager.delete_syftbox(verbose=False, broadcast_delete_events=False)

    keys_path = datasite_crypto_keys_path(do_manager.syftbox_folder, do_manager.email)
    reloaded = PeerStore.create(email=do_manager.email, keys_path=keys_path)
    assert reloaded.get_public_bundle() == store.get_public_bundle()


def test_plaintext_pair_publishes_bundles_at_peering():
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection()
    ds_router = ds_manager.peer_manager.connection_router
    do_router = do_manager.peer_manager.connection_router
    assert ds_router.read_peer_encryption_bundle(do_manager.email) is not None
    assert do_router.read_peer_encryption_bundle(ds_manager.email) is not None


def _publish_bundle_of_other_keys(do_manager, ds_manager) -> None:
    """As a new runtime without the key file: Drive holds the bundle of older keys."""
    earlier = PeerStore(email=do_manager.email)
    earlier.generate_keys()
    do_manager.peer_manager.connection_router.write_encryption_bundle(
        ds_manager.email,
        json.dumps({"public_encryption_bundle": earlier.get_public_bundle()}),
    )
    do_manager.peer_manager._states_at_start = None  # as a new process


def test_first_peer_load_refuses_bundle_of_other_keys():
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        do_encryption_policy=REQUIRED,
    )
    _publish_bundle_of_other_keys(do_manager, ds_manager)
    with pytest.raises(ValueError, match="already has a key"):
        do_manager.load_peers()


def test_bundle_of_other_keys_ignored_when_pair_does_not_encrypt():
    """No peer uses the bundle, so new keys on a new runtime are fine."""
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection()
    _publish_bundle_of_other_keys(do_manager, ds_manager)
    do_manager.load_peers()


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


def test_message_from_peer_with_unknown_policy_refused():
    _, bob = _keyed_pair(OPTIONAL, OPTIONAL)
    bob.get_cached_peer("alice@example.com").version = None
    with pytest.raises(SyftPeerNotReadyError, match="not known"):
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


def test_receiver_rejects_envelope_when_pair_does_not_encrypt():
    alice, bob = _keyed_pair(REQUIRED, OPTIONAL)
    envelope = alice.encrypt("bob@example.com", b"secret")
    bob.get_cached_peer("alice@example.com").version = _version_with_policy(OPTIONAL)
    with pytest.raises(MessageFormatError, match="is encrypted"):
        bob.decrypt_and_verify_if_needed("alice@example.com", envelope)


def test_envelope_without_peer_bundle_is_not_ready():
    alice, bob = _keyed_pair(REQUIRED, OPTIONAL)
    envelope = alice.encrypt("bob@example.com", b"secret")
    bob.get_cached_peer("alice@example.com").public_encryption_bundle = None
    with pytest.raises(SyftPeerNotReadyError):
        bob.decrypt_and_verify_if_needed("alice@example.com", envelope)


# =========================================================================
# End to end over the mock drive
# =========================================================================


@pytest.mark.parametrize(
    "ds_policy, do_policy",
    [(OPTIONAL, REQUIRED), (REQUIRED, OPTIONAL)],
)
def test_negotiated_encryption_flow(ds_policy, do_policy):
    """The optional side complies with no manual configuration."""
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        ds_encryption_policy=ds_policy,
        do_encryption_policy=do_policy,
    )
    grant_job_inbox_access(do_manager, ds_manager.email)
    # One load_peers round each lets both sides see the policy and read bundles.
    ds_manager.load_peers()
    do_manager.load_peers()

    file_path = path_for_job(do_manager.email, ds_manager.email, "my.job")
    ds_manager._send_file_change(file_path, "negotiated content")
    assert PeerStore.is_envelope(_raw_inbox_message(do_manager, ds_manager.email))

    do_manager.sync()
    assert any("negotiated content" in c for c in _accepted_contents(do_manager))

    ds_manager.sync()
    assert len(_ds_cached_contents(ds_manager)) > 0


def test_owner_send_to_non_peer_raises():
    _, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        do_encryption_policy=REQUIRED,
    )
    syncer = do_manager.datasite_owner_syncer
    message = FileChangeEventsMessage(events=[])
    syncer.outbox_queue.put(("former@example.com", message))
    with pytest.raises(SyftPeerNotReadyError, match="not known"):
        syncer.process_syftbox_events_queue()


def test_owner_leaves_envelope_from_peer_without_bundle_in_inbox():
    """The sync goes on; the message stays in the inbox until the bundle is read."""
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        ds_encryption_policy=REQUIRED,
        do_encryption_policy=OPTIONAL,
    )
    grant_job_inbox_access(do_manager, ds_manager.email)
    do_manager.load_peers()
    file_path = path_for_job(do_manager.email, ds_manager.email, "my.job")
    ds_manager._send_file_change(file_path, "bundle later")

    peer = do_manager.peer_manager.peer_store.get_cached_peer(ds_manager.email)
    peer.public_encryption_bundle = None
    with pytest.warns(UserWarning, match="stay in the inbox"):
        do_manager.datasite_owner_syncer.sync([ds_manager.email])
    assert not any("bundle later" in c for c in _accepted_contents(do_manager))

    do_manager.sync()
    assert any("bundle later" in c for c in _accepted_contents(do_manager))


# =========================================================================
# Sends that cannot go
# =========================================================================


def _queued_changes(manager) -> int:
    return manager.datasite_watcher_syncer.queue.qsize()


@pytest.mark.parametrize("do_policy", [OPTIONAL, REQUIRED])
def test_send_before_owner_accepts_raises(do_policy):
    """Before the owner accepts, its policy is not known, so nothing is sent."""
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        add_peers=False,
        do_encryption_policy=do_policy,
    )
    ds_manager.add_peer(do_manager.email, sync=False)

    file_path = path_for_job(do_manager.email, ds_manager.email, "my.job")
    with pytest.raises(SyftPeerNotReadyError, match="not known"):
        ds_manager._send_file_change(file_path, "early content")
    assert _queued_changes(ds_manager) == 0

    do_manager.load_peers()
    do_manager.approve_peer_request(ds_manager.email)
    grant_job_inbox_access(do_manager, ds_manager.email)
    do_manager.sync()
    assert not any("early content" in c for c in _accepted_contents(do_manager))


def test_send_to_non_peer_raises():
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        add_peers=False,
    )
    file_path = path_for_job(do_manager.email, ds_manager.email, "my.job")
    with pytest.raises(SyftPeerNotReadyError, match="not known"):
        ds_manager._send_file_change(file_path, "content")


def test_changes_for_two_datasites_go_to_each(monkeypatch):
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection()
    sent = []
    monkeypatch.setattr(
        ConnectionRouter,
        "watcher_send_proposed_file_changes_message",
        lambda self, recipient, message: (
            sent.append(
                (recipient, {fc.datasite_email for fc in message.proposed_file_changes})
            )
        ),
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


def _not_ready(recipient: str) -> SyftPeerNotReadyError:
    return SyftPeerNotReadyError(recipient, cause="No bundle.", remedy="Sync.")


def test_failure_for_one_recipient_still_sends_to_other(monkeypatch):
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection()
    sent = []

    def send_message(self, recipient, message):
        if recipient == "other@example.com":
            raise _not_ready(recipient)
        sent.append(recipient)

    monkeypatch.setattr(
        ConnectionRouter, "watcher_send_proposed_file_changes_message", send_message
    )
    watcher = ds_manager.datasite_watcher_syncer
    watcher.on_file_change("other@example.com/a.txt", "a", process_now=False)
    watcher.on_file_change(f"{do_manager.email}/b.txt", "b", process_now=False)
    with pytest.raises(SyftPeerNotReadyError):
        watcher.process_file_changes_queue()
    assert sent == [do_manager.email]
    assert _queued_changes(ds_manager) == 0


def test_unsent_owner_change_stays_pending(monkeypatch):
    """The send raises, and the next sync sends the change."""
    ds_manager, do_manager = _pair_with_unsent_do_file()

    def fail(self, recipient, message):
        raise _not_ready(recipient)

    with monkeypatch.context() as m:
        m.setattr(ConnectionRouter, "owner_write_event_messages_to_outbox", fail)
        with pytest.raises(SyftPeerNotReadyError):
            do_manager.sync(auto_checkpoint=False)

    _assert_ds_receives_after_notebook_sync(ds_manager, do_manager)


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


def test_plaintext_outbox_message_skipped_and_later_ones_read(monkeypatch):
    ds_manager, do_manager = _required_ds_pair()
    _owner_answers_in_plaintext(ds_manager, do_manager, "plain content", monkeypatch)

    file_path = path_for_job(do_manager.email, ds_manager.email, "other.job")
    ds_manager._send_file_change(file_path, "encrypted content")
    do_manager.datasite_owner_syncer.sync([ds_manager.email])
    with pytest.warns(UserWarning, match="not encrypted"):
        ds_manager.sync()
    contents = _ds_cached_contents(ds_manager)
    assert not any("plain content" in c for c in contents)
    assert any("encrypted content" in c for c in contents)
