import pytest

import syft.sync.syftbox_manager as syftbox_manager_module
from syft.sync.connections.drive import mock_drive_service
from syft.sync.peers.peer import PeerSetupError
from syft.sync.syftbox_manager import SyftboxManager
from tests.unit.test_validate_peer import _pair, _start_both_then_request


def _no_sleep(_seconds):
    raise AssertionError("wait_until_peered must not wait here")


def test_wait_until_peered_waits_for_late_approval(monkeypatch):
    ds_manager, do_manager = _pair(add_peers=False)
    ds_manager.add_peer(do_manager.email)
    do_manager.load_peers()
    waits = []

    def approve_during_wait(seconds):
        waits.append(seconds)
        do_manager.approve_peer_request(ds_manager.email)

    monkeypatch.setattr(syftbox_manager_module.time, "sleep", approve_during_wait)

    peer = ds_manager.wait_until_peered(do_manager.email, timeout=60, poll_interval=1)
    assert peer.is_approved
    assert waits == [1]


def test_wait_until_peered_fails_at_once_when_waiting_cannot_help(monkeypatch):
    ds_manager, do_manager = _pair(add_peers=False)
    _start_both_then_request(ds_manager, do_manager)
    monkeypatch.setattr(syftbox_manager_module.time, "sleep", _no_sleep)

    with pytest.raises(PeerSetupError, match="not a peer"):
        ds_manager.wait_until_peered("wrong@test.com", timeout=60)
    # A request that only this client can approve.
    with pytest.raises(PeerSetupError, match="approve_peer_request"):
        do_manager.wait_until_peered(ds_manager.email, timeout=60)


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


def test_wait_until_peered_polls_with_one_drive_request(monkeypatch):
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

    with pytest.raises(TimeoutError, match="not approved"):
        ds_manager.wait_until_peered(do_manager.email, timeout=4, poll_interval=1)
    # Four polls in four seconds. Before the first sleep: the initial full
    # load. After it: one Drive request per poll, and no other load.
    assert len(per_poll) == 4
    assert per_poll[1:] == [1, 1, 1]
    assert drive_requests[0] == 1
    assert loads[0] == 1
