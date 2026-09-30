import inspect
from datetime import datetime, timedelta, timezone

import pytest

import syft.sync.syftbox_manager as syftbox_manager_module
from syft.sync.connections.drive import mock_drive_service
from syft.sync.peers.peer import PeerNotReadyError, PeerSetupError
from syft.sync.syftbox_manager import SyftboxManager


def _pair(add_peers: bool):
    return SyftboxManager.pair_with_mock_drive_service_connection(
        email1="do@test.com", email2="ds@test.com", add_peers=add_peers
    )


def _set_last_login(manager: SyftboxManager, when: datetime) -> None:
    """Make Drive report that ``manager`` last wrote its version file at ``when``."""
    connection = manager._connection_router.connection_for_own_syftbox()
    store = connection.drive_service._backing_store
    store.files[connection._get_version_file_id()].modifiedTime = (
        when.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    )


def _no_sleep(_seconds):
    raise AssertionError("validate_peer must not wait here")


def test_peer_connected_in_this_session_is_valid():
    ds_manager, do_manager = _pair(add_peers=False)
    ds_manager.add_peer(do_manager.email)
    do_manager.load_peers()
    do_manager.approve_peer_request(ds_manager.email)

    assert ds_manager.validate_peer(do_manager.email).email == do_manager.email
    assert do_manager.validate_peer(ds_manager.email).email == ds_manager.email


def test_unknown_peer_is_invalid():
    ds_manager, _ = _pair(add_peers=True)

    with pytest.raises(PeerSetupError, match="not a peer"):
        ds_manager.validate_peer("wrong@test.com")


def test_unapproved_peer_is_invalid():
    ds_manager, do_manager = _pair(add_peers=False)
    ds_manager.add_peer(do_manager.email)

    with pytest.raises(PeerNotReadyError, match="not approved"):
        ds_manager.validate_peer(do_manager.email)


def test_peer_request_not_approved_by_me_is_invalid():
    ds_manager, do_manager = _pair(add_peers=False)
    ds_manager.add_peer(do_manager.email)

    with pytest.raises(PeerSetupError, match="approve_peer_request"):
        do_manager.validate_peer(ds_manager.email)


def test_peer_approved_without_request_is_invalid_until_peer_connects():
    # approve_peer_request(..., peer_must_exist=False) with a wrong email only
    # sends a request; the peer is not connected.
    ds_manager, do_manager = _pair(add_peers=False)
    do_manager.approve_peer_request(ds_manager.email, peer_must_exist=False)

    with pytest.raises(PeerNotReadyError, match="not approved"):
        do_manager.validate_peer(ds_manager.email)

    ds_manager.add_peer(do_manager.email)
    assert do_manager.validate_peer(ds_manager.email).is_approved


def test_leftover_connection_with_stale_peer_is_invalid():
    ds_manager, do_manager = _pair(add_peers=True)
    # The DS loaded peers first after the DO approved: the connection was
    # already there when the DS client started.
    _set_last_login(do_manager, datetime.now(timezone.utc) - timedelta(days=2))

    with pytest.raises(PeerNotReadyError, match="already connected"):
        ds_manager.validate_peer(do_manager.email)


def test_leftover_connection_with_same_day_peer_login_is_valid():
    ds_manager, do_manager = _pair(add_peers=True)
    _set_last_login(do_manager, datetime.now(timezone.utc) - timedelta(hours=2))

    assert ds_manager.validate_peer(do_manager.email).is_approved


def test_login_time_comes_from_drive_not_from_the_peer_clock():
    ds_manager, do_manager = _pair(add_peers=True)
    # The peer's clock is two days behind; Drive stamps the upload with its own.
    do_manager.peer_manager.get_own_version().updated_at = datetime.now(
        timezone.utc
    ) - timedelta(days=2)
    do_manager.peer_manager.write_own_version()

    assert ds_manager.validate_peer(do_manager.email).is_approved


def test_max_age_sets_how_recent_the_peer_login_must_be():
    ds_manager, do_manager = _pair(add_peers=True)
    _set_last_login(do_manager, datetime.now(timezone.utc) - timedelta(hours=2))

    assert ds_manager.validate_peer(do_manager.email, max_age=timedelta(hours=3))
    with pytest.raises(PeerNotReadyError, match="already connected"):
        ds_manager.validate_peer(do_manager.email, max_age=timedelta(hours=1))


def test_timeout_waits_for_a_late_approval(monkeypatch):
    ds_manager, do_manager = _pair(add_peers=False)
    ds_manager.add_peer(do_manager.email)
    do_manager.load_peers()
    waits = []

    def approve_during_wait(seconds):
        waits.append(seconds)
        do_manager.approve_peer_request(ds_manager.email)

    monkeypatch.setattr(syftbox_manager_module.time, "sleep", approve_during_wait)

    peer = ds_manager.validate_peer(do_manager.email, timeout=60, poll_interval=1)
    assert peer.is_approved
    assert waits == [1]


def test_timeout_expires_when_the_peer_never_approves():
    ds_manager, do_manager = _pair(add_peers=False)
    ds_manager.add_peer(do_manager.email)

    with pytest.raises(PeerNotReadyError, match="not approved"):
        ds_manager.validate_peer(do_manager.email, timeout=0.05, poll_interval=0.01)


def test_timeout_does_not_wait_when_waiting_cannot_help(monkeypatch):
    ds_manager, do_manager = _pair(add_peers=False)
    ds_manager.add_peer(do_manager.email)
    monkeypatch.setattr(syftbox_manager_module.time, "sleep", _no_sleep)

    with pytest.raises(PeerSetupError, match="not a peer"):
        ds_manager.validate_peer("wrong@test.com", timeout=60)
    # A request that only this client can approve.
    with pytest.raises(PeerSetupError, match="approve_peer_request"):
        do_manager.validate_peer(ds_manager.email, timeout=60)


def _count_drive_requests(monkeypatch) -> list[int]:
    """Count mock Drive requests; the returned one-item list holds the count."""
    count = [0]
    for request_class in (
        mock_drive_service.MockListRequest,
        mock_drive_service.MockGetRequest,
    ):
        original = request_class.execute

        def counted(self, *args, _original=original, **kwargs):
            count[0] += 1
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(request_class, "execute", counted)
    # Downloads go through get_media(), not execute().
    original_get_media = mock_drive_service.MockFilesResource.get_media

    def counted_get_media(self, *args, **kwargs):
        count[0] += 1
        return original_get_media(self, *args, **kwargs)

    monkeypatch.setattr(
        mock_drive_service.MockFilesResource, "get_media", counted_get_media
    )
    return count


def _count_load_peers(monkeypatch) -> list[int]:
    count = [0]
    original = SyftboxManager.load_peers

    def counted(self, *args, **kwargs):
        count[0] += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(SyftboxManager, "load_peers", counted)
    return count


def test_wait_for_approval_polls_with_one_drive_request(monkeypatch):
    ds_manager, do_manager = _pair(add_peers=False)
    ds_manager.add_peer(do_manager.email)
    drive_requests = _count_drive_requests(monkeypatch)
    loads = _count_load_peers(monkeypatch)
    clock = [0.0]
    per_poll = []

    def record_poll(seconds):
        clock[0] += seconds
        per_poll.append(drive_requests[0])
        drive_requests[0] = 0

    monkeypatch.setattr(syftbox_manager_module.time, "sleep", record_poll)
    monkeypatch.setattr(syftbox_manager_module.time, "monotonic", lambda: clock[0])

    with pytest.raises(PeerNotReadyError, match="not approved"):
        ds_manager.validate_peer(do_manager.email, timeout=4, poll_interval=1)
    # Four polls in four seconds. Before the first sleep: the initial full
    # load. After it: one Drive request per poll, and no other load.
    assert len(per_poll) == 4
    assert per_poll[1:] == [1, 1, 1]
    assert drive_requests[0] == 1
    assert loads[0] == 1


def test_wait_for_login_reloads_once_the_peer_logs_in(monkeypatch):
    ds_manager, do_manager = _pair(add_peers=True)
    _set_last_login(do_manager, datetime.now(timezone.utc) - timedelta(days=2))
    loads = _count_load_peers(monkeypatch)
    polls = []

    def log_in_on_second_poll(_seconds):
        polls.append(_seconds)
        if len(polls) == 2:
            do_manager.peer_manager.write_own_version()

    monkeypatch.setattr(syftbox_manager_module.time, "sleep", log_in_on_second_poll)

    assert ds_manager.validate_peer(do_manager.email, timeout=60, poll_interval=1)
    assert len(polls) == 2
    assert loads[0] == 2


def test_default_poll_interval_is_at_least_15s():
    default = (
        inspect.signature(SyftboxManager.validate_peer)
        .parameters["poll_interval"]
        .default
    )
    assert default >= 15


def test_timeout_shorter_than_poll_interval_still_waits(monkeypatch):
    ds_manager, do_manager = _pair(add_peers=False)
    ds_manager.add_peer(do_manager.email)
    do_manager.load_peers()
    clock = [0.0]
    sleeps = []

    def approve_during_wait(seconds):
        clock[0] += seconds
        sleeps.append(seconds)
        do_manager.approve_peer_request(ds_manager.email)

    monkeypatch.setattr(syftbox_manager_module.time, "sleep", approve_during_wait)
    monkeypatch.setattr(syftbox_manager_module.time, "monotonic", lambda: clock[0])

    peer = ds_manager.validate_peer(do_manager.email, timeout=10, poll_interval=15)
    assert peer.is_approved
    assert sleeps == [10]
