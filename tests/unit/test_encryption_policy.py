"""Per-peer encryption negotiated from each side's advertised policy.

A client is ``optional`` (encrypt only when the peer requires it) or
``required``. A pair encrypts when either side requires it.
"""

import json

import pytest
from pydantic import ValidationError

from syft.sync.connections.drive.drive_peering import DrivePeeringConnection
from syft.sync.connections.connection_router import ConnectionRouter
from syft.sync.connections.drive import mock_drive_service
from syft.sync.encryption_policy import (
    EncryptionPolicy,
    EncryptionPolicyMismatchError,
    MessageFormatError,
)
from syft.sync.events.file_change_event import (
    FileChangeEventsMessage,
)
from syft.sync.peers.exceptions import SyftPeerNotReadyError
from syft.sync.peers.peer import Peer
from syft.sync.peers.peer_store import (
    PeerStore,
    datasite_settings_path,
    resolve_encryption_policy,
)
from syft.sync.syftbox_manager import SyftboxManager
from syft.sync.sync.datasite_owner_syncer import OutgoingMessage
from syft.sync.connections.types import MessageKind
from syft.sync.version.peer_info import PeerInfo
from syft.sync.version.peer_manager import PeerManager, PeerManagerConfig
from syft.sync.version.version_info import VersionInfo
from tests.unit.test_persisted_cache import (
    _assert_ds_receives_after_notebook_sync,
    _pair_with_unsent_do_file,
)
from tests.unit.test_sync_manager import path_for_job
from tests.unit.utils import grant_job_inbox_access

OPTIONAL = EncryptionPolicy.OPTIONAL
REQUIRED = EncryptionPolicy.REQUIRED


def _peer_info(policy: EncryptionPolicy | None) -> PeerInfo | None:
    """The info a peer publishes, or None when it was not read yet."""
    if policy is None:
        return None
    return PeerInfo(version_info=VersionInfo.current(), encryption_policy=policy)


def _store_with_peer(
    my_policy: EncryptionPolicy, peer_policy: EncryptionPolicy | None
) -> PeerStore:
    store = PeerStore(email="alice@example.com", encryption_policy=my_policy)
    store.add_peer(Peer(email="bob@example.com", info=_peer_info(peer_policy)))
    return store


def _keyed_pair(
    alice_policy: EncryptionPolicy, bob_policy: EncryptionPolicy
) -> tuple[PeerStore, PeerStore]:
    """Two stores that know each other's policy and hold each other's bundle."""
    alice = PeerStore(email="alice@example.com", encryption_policy=alice_policy)
    bob = PeerStore(email="bob@example.com", encryption_policy=bob_policy)
    alice.generate_keys()
    bob.generate_keys()
    alice.add_peer(Peer(email="bob@example.com", info=_peer_info(bob_policy)))
    bob.add_peer(Peer(email="alice@example.com", info=_peer_info(alice_policy)))
    alice.set_peer_bundle("bob@example.com", bob.get_public_bundle())
    bob.set_peer_bundle("alice@example.com", alice.get_public_bundle())
    return alice, bob


def _raw_inbox_message(do_manager, sender_email: str) -> bytes:
    router = do_manager.datasite_owner_syncer.connection_router
    [message] = router.connections.p2p_con.receive(
        sender_email, MessageKind.PROPOSED_CHANGES, max_messages=1
    )
    return message.data


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
    stale.info = _peer_info(OPTIONAL)
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


def test_no_encryption_setting_maps_to_no_policy():
    assert EncryptionPolicy.from_bool(None) is None


@pytest.mark.parametrize("value", [OPTIONAL, "optional", 1])
def test_encryption_setting_must_be_bool_or_none(value):
    with pytest.raises(TypeError, match="True, False or None"):
        EncryptionPolicy.from_bool(value)


def test_own_version_advertises_policy():
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        ds_encryption_policy=OPTIONAL,
        do_encryption_policy=REQUIRED,
    )
    assert do_manager.peer_manager.get_own_peer_info().encryption_policy == REQUIRED
    assert ds_manager.peer_manager.get_own_peer_info().encryption_policy == OPTIONAL
    advertised = ds_manager.peer_manager.get_cached_peer(do_manager.email)
    assert advertised.encryption_policy == REQUIRED


def test_missing_version_file_created_with_own_policy():
    a_cached_client, b_cached_client = mock_drive_service.pair_with_mock_service(
        "a@example.com", "b@example.com"
    )
    DrivePeeringConnection(a_cached_client).grant_peer_info_read(
        "b@example.com", _peer_info(REQUIRED)
    )
    assert (
        DrivePeeringConnection(b_cached_client)
        .read_peer_info("a@example.com")
        .encryption_policy
        == REQUIRED
    )


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
    assert store.pair_encrypts(store.get_cached_peer("bob@example.com")) is encrypted


def test_resolution_follows_changed_peer_policy():
    """An enclave that reboots with another setting is picked up on the next load."""
    store = _store_with_peer(OPTIONAL, OPTIONAL)
    assert not store.pair_encrypts(store.get_cached_peer("bob@example.com"))

    store.get_cached_peer("bob@example.com").info = _peer_info(REQUIRED)
    assert store.pair_encrypts(store.get_cached_peer("bob@example.com"))

    store.get_cached_peer("bob@example.com").info = _peer_info(OPTIONAL)
    assert not store.pair_encrypts(store.get_cached_peer("bob@example.com"))


def test_sender_refuses_peer_with_unknown_policy():
    store = PeerStore(email="alice@example.com", encryption_policy=OPTIONAL)
    store.add_peer(Peer(email="bob@example.com"))
    with pytest.raises(SyftPeerNotReadyError, match="not known"):
        store.encrypt_if_needed("bob@example.com", b"data")


def test_required_sender_encrypts_without_peer_policy():
    """A client that requires encryption encrypts whatever the peer advertises."""
    store = PeerStore(email="alice@example.com", encryption_policy=REQUIRED)
    store.generate_keys()
    store.add_peer(Peer(email="bob@example.com"))
    with pytest.raises(SyftPeerNotReadyError, match="public encryption bundle"):
        store.encrypt_if_needed("bob@example.com", b"data")


def test_unread_peer_info_policy_not_known():
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


@pytest.mark.parametrize(
    "requested, expected", [(None, OPTIONAL), (REQUIRED, REQUIRED)]
)
def test_first_client_writes_policy(tmp_path, requested, expected):
    path = tmp_path / "syft_settings.json"
    assert resolve_encryption_policy(path, requested) == expected
    assert resolve_encryption_policy(path, None) == expected


def test_client_with_other_policy_than_disk_raises(tmp_path):
    path = tmp_path / "syft_settings.json"
    resolve_encryption_policy(path, OPTIONAL)
    with pytest.raises(EncryptionPolicyMismatchError, match="'optional'"):
        resolve_encryption_policy(path, REQUIRED)


def test_client_without_policy_takes_policy_of_syftbox(tmp_path):
    """syft-bg starts after a notebook that required encryption."""

    def start(policy):
        config = PeerManagerConfig(
            syftbox_folder=tmp_path, email="alice@example.com", encryption_policy=policy
        )
        return PeerManager.peer_store_from_config(config).encryption_policy

    assert start(REQUIRED) == REQUIRED
    assert start(None) == REQUIRED


def test_sync_raises_when_policy_on_disk_changed():
    """Another process deleted the SyftBox and started with another policy."""
    _, do_manager = SyftboxManager.pair_with_mock_drive_service_connection()
    settings = datasite_settings_path(do_manager.syftbox_folder, do_manager.email)
    settings.unlink()
    resolve_encryption_policy(settings, REQUIRED)
    with pytest.raises(EncryptionPolicyMismatchError):
        do_manager.sync()


def test_policy_check_writes_missing_settings_back():
    """After delete_syftbox, the running client keeps its policy for the SyftBox."""
    _, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        do_encryption_policy=REQUIRED
    )
    settings = datasite_settings_path(do_manager.syftbox_folder, do_manager.email)
    settings.unlink()
    do_manager.peer_manager.check_encryption_policy()
    assert resolve_encryption_policy(settings, None) == REQUIRED


@pytest.mark.parametrize("policy", [OPTIONAL, REQUIRED])
def test_every_policy_creates_key_file(tmp_path, policy):
    keys_path = tmp_path / "crypto_keys.json"
    store = PeerStore.create(
        email="alice@example.com", encryption_policy=policy, keys_path=keys_path
    )
    reloaded = PeerStore.create(email="alice@example.com", keys_path=keys_path)
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
    do_manager.peer_manager._first_load_done = False  # as a new process


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
    bob.get_cached_peer("alice@example.com").info = None
    with pytest.raises(SyftPeerNotReadyError, match="not known"):
        bob.decrypt_and_verify_if_needed("alice@example.com", b"plain")


def test_receiver_accepts_plaintext_in_plaintext_pair():
    _, bob = _keyed_pair(OPTIONAL, OPTIONAL)
    assert bob.decrypt_and_verify_if_needed("alice@example.com", b"plain") == b"plain"


def test_receiver_rejects_envelope_when_pair_does_not_encrypt():
    alice, bob = _keyed_pair(REQUIRED, OPTIONAL)
    envelope = alice.encrypt("bob@example.com", b"secret")
    bob.get_cached_peer("alice@example.com").info = _peer_info(OPTIONAL)
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


def test_owner_send_to_non_peer_warns():
    _, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        do_encryption_policy=REQUIRED,
    )
    syncer = do_manager.datasite_owner_syncer
    message = FileChangeEventsMessage(events=[])
    syncer.outbox_queue.put(OutgoingMessage("former@example.com", message))
    with pytest.warns(UserWarning, match="not known"):
        syncer._send_outbox_messages()


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
    """The send warns, and the next sync sends the change."""
    ds_manager, do_manager = _pair_with_unsent_do_file()

    def fail(self, recipient, message):
        raise _not_ready(recipient)

    with monkeypatch.context() as m:
        m.setattr(ConnectionRouter, "owner_write_event_messages_to_outbox", fail)
        with pytest.warns(UserWarning, match="are not sent"):
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
