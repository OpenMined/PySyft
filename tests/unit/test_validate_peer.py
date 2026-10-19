import pytest

from syft.sync.peers.peer import PeerNotReadyError, PeerSetupError
from syft.sync.syftbox_manager import SyftboxManager


def _pair(add_peers: bool):
    return SyftboxManager.pair_with_mock_drive_service_connection(
        email1="do@test.com", email2="ds@test.com", add_peers=add_peers
    )


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


def test_leftover_connection_warns_and_passes():
    ds_manager, do_manager = _pair(add_peers=True)
    # The DS loaded peers first after the DO approved: the connection was
    # already there when the DS client started.

    with pytest.warns(UserWarning, match="already connected before this client"):
        peer = ds_manager.validate_peer(do_manager.email)
    assert peer.is_approved


def test_request_waiting_at_start_warns_once_approved():
    # The DS request was already there when the DO client started.
    ds_manager, do_manager = _pair(add_peers=False)
    ds_manager.add_peer(do_manager.email)

    with pytest.raises(PeerSetupError, match="approve_peer_request"):
        do_manager.validate_peer(ds_manager.email)
    do_manager.approve_peer_request(ds_manager.email)
    with pytest.warns(UserWarning, match="sent its request before this client"):
        assert do_manager.validate_peer(ds_manager.email).is_approved


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
