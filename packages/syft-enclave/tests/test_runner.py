"""Tests for EnclaveRunner — focused on fresh_state init behavior."""

from unittest.mock import MagicMock

from syft_enclaves.runner import EnclaveRunner


def _make_client():
    """Build a stub SyftEnclaveClient just deep enough for the init phases."""
    client = MagicMock()
    client.email = "enclave@openmined.org"
    client.syftbox_folder = "/tmp/SyftBox_enclave"
    # _on_peering needs these
    client.peers = []
    client.data_owners = []
    return client


def _make_peer(email, state="requested_by_peer"):
    peer = MagicMock()
    peer.email = email
    peer.state = state
    return peer


def test_accept_peers_approves_only_data_owners():
    """Peer requests from emails outside data_owners are ignored."""
    client = _make_client()
    client.data_owners = ["do1@x.com", "DO2@Y.com"]
    client.peers = [
        _make_peer("do1@x.com"),
        _make_peer("do2@y.com"),  # allowlist match is case-insensitive
        _make_peer("stranger@evil.com"),
    ]
    runner = EnclaveRunner(client=client)

    runner._accept_peers()

    approved = [c.args[0] for c in client.approve_peer_request.call_args_list]
    assert approved == ["do1@x.com", "do2@y.com"]


def test_accept_peers_empty_allowlist_accepts_none():
    """No data owners configured — no peer is ever accepted."""
    client = _make_client()
    client.peers = [_make_peer("anyone@x.com")]
    runner = EnclaveRunner(client=client)

    runner._accept_peers()

    client.approve_peer_request.assert_not_called()


def test_fresh_state_true_invokes_delete_syftbox(tmp_path, monkeypatch):
    """With fresh_state=True (default), _on_initializing must wipe state once."""
    # Make _on_attesting a no-op (no TEE socket present in unit tests).
    monkeypatch.setattr(
        "syft_enclaves.runner.TEE_SOCKET_PATH", tmp_path / "nonexistent"
    )

    client = _make_client()
    runner = EnclaveRunner(client=client, fresh_state=True)
    runner.init()

    client._manager.delete_syftbox.assert_called_once_with()


def test_fresh_state_false_skips_delete_syftbox(tmp_path, monkeypatch):
    """Opting out (fresh_state=False) preserves state across init."""
    monkeypatch.setattr(
        "syft_enclaves.runner.TEE_SOCKET_PATH", tmp_path / "nonexistent"
    )

    client = _make_client()
    runner = EnclaveRunner(client=client, fresh_state=False)
    runner.init()

    client._manager.delete_syftbox.assert_not_called()


def test_fresh_state_default_is_true(tmp_path, monkeypatch):
    """Constructor default unified with settings default — fresh state on by default."""
    monkeypatch.setattr(
        "syft_enclaves.runner.TEE_SOCKET_PATH", tmp_path / "nonexistent"
    )

    client = _make_client()
    runner = EnclaveRunner(client=client)  # no fresh_state arg
    assert runner.fresh_state is True
    runner.init()
    client._manager.delete_syftbox.assert_called_once_with()


def test_fresh_state_uses_default_kwargs_on_delete(tmp_path, monkeypatch):
    """We rely on delete_syftbox's own defaults — no kwargs passed."""
    monkeypatch.setattr(
        "syft_enclaves.runner.TEE_SOCKET_PATH", tmp_path / "nonexistent"
    )

    client = _make_client()
    EnclaveRunner(client=client, fresh_state=True).init()

    # Must be called with no positional or keyword args — let the method's
    # own defaults handle broadcast_delete_events and verbose.
    call = client._manager.delete_syftbox.call_args
    assert call.args == ()
    assert call.kwargs == {}
