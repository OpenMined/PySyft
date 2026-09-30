import pytest

import syft.sync.syftbox_manager as syftbox_manager_module
from syft.sync.connections.drive import mock_drive_service
from syft.sync.peers.peer import PeerNotReadyError, PeerSetupError
from syft.sync.syftbox_manager import SyftboxManager


def _pair(add_peers: bool):
    return SyftboxManager.pair_with_mock_drive_service_connection(
        email1="do@test.com", email2="ds@test.com", add_peers=add_peers
    )


def _no_sleep(_seconds):
    raise AssertionError("validate_peer must not wait here")


def _start_both_then_request(ds_manager, do_manager):
    """Both clients load their peers, then the DS sends its request."""
    do_manager.load_peers()
    ds_manager.add_peer(do_manager.email)


def test_peer_connected_in_this_session_is_valid():
    ds_manager, do_manager = _pair(add_peers=False)
    _start_both_then_request(ds_manager, do_manager)
    do_manager.approve_peer_request(ds_manager.email)

    assert ds_manager.validate_peer(do_manager.email).email == do_manager.email
    assert do_manager.validate_peer(ds_manager.email).email == ds_manager.email


def test_unapproved_peer_is_invalid():
    ds_manager, do_manager = _pair(add_peers=False)
    ds_manager.add_peer(do_manager.email)

    with pytest.raises(PeerNotReadyError, match="not approved"):
        ds_manager.validate_peer(do_manager.email)


def test_peer_approved_without_request_is_invalid_until_peer_connects():
    # approve_peer_request(..., peer_must_exist=False) with a wrong email only
    # sends a request; the peer is not connected.
    ds_manager, do_manager = _pair(add_peers=False)
    do_manager.approve_peer_request(ds_manager.email, peer_must_exist=False)

    with pytest.raises(PeerNotReadyError, match="not approved"):
        do_manager.validate_peer(ds_manager.email)

    ds_manager.add_peer(do_manager.email)
    assert do_manager.validate_peer(ds_manager.email).is_approved


def test_leftover_connection_warns_and_passes(monkeypatch):
    ds_manager, do_manager = _pair(add_peers=True)
    monkeypatch.setattr(syftbox_manager_module.time, "sleep", _no_sleep)
    # The DS loaded peers first after the DO approved: the connection was
    # already there when the DS client started.

    with pytest.warns(UserWarning, match="already connected before this client"):
        peer = ds_manager.validate_peer(do_manager.email, timeout=60)
    assert peer.is_approved


def test_request_waiting_at_start_warns_once_approved(monkeypatch):
    # The DS request was already there when the DO client started.
    ds_manager, do_manager = _pair(add_peers=False)
    ds_manager.add_peer(do_manager.email)
    monkeypatch.setattr(syftbox_manager_module.time, "sleep", _no_sleep)

    with pytest.raises(PeerSetupError, match="approve_peer_request"):
        do_manager.validate_peer(ds_manager.email, timeout=60)
    do_manager.approve_peer_request(ds_manager.email)
    with pytest.warns(UserWarning, match="sent its request before this client"):
        assert do_manager.validate_peer(ds_manager.email).is_approved


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


def test_timeout_does_not_wait_when_waiting_cannot_help(monkeypatch):
    ds_manager, do_manager = _pair(add_peers=False)
    _start_both_then_request(ds_manager, do_manager)
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


def test_reset_then_re_peer_in_the_same_session_is_valid(recwarn):
    ds_manager, do_manager = _pair(add_peers=True)  # connected at start
    for manager in (ds_manager, do_manager):
        manager.delete_syftbox(verbose=False, broadcast_delete_events=False)
        manager.peer_manager.write_own_version()

    ds_manager.add_peer(do_manager.email)
    do_manager.approve_peer_request(ds_manager.email, peer_must_exist=False)

    assert ds_manager.validate_peer(do_manager.email).is_approved
    assert do_manager.validate_peer(ds_manager.email).is_approved
    assert not [w for w in recwarn if "before this client started" in str(w.message)]
