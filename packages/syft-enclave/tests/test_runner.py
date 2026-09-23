"""Tests for EnclaveRunner — focused on fresh_state init behavior."""

from unittest.mock import MagicMock

from attestation_helpers import FAKE_KEY_FINGERPRINT

from syft_enclaves.runner import EnclaveRunner


def _make_client():
    """Build a stub SyftEnclaveClient just deep enough for the init phases."""
    client = MagicMock()
    client.email = "enclave@openmined.org"
    client.syftbox_folder = "/tmp/SyftBox_enclave"
    # _on_peering needs these
    client.peers = []
    return client


def test_fresh_state_true_invokes_delete_syftbox(tmp_path, monkeypatch):
    """With fresh_state=True (default), _on_initializing must wipe state once."""
    # Make _on_attesting a no-op (no TEE socket present in unit tests).
    monkeypatch.setattr(
        "syft_enclaves.runner.TEE_SOCKET_PATH", tmp_path / "nonexistent"
    )

    client = _make_client()
    runner = EnclaveRunner(client=client, fresh_state=True)
    runner.init()

    client.delete_syftbox.assert_called_once_with()


def test_fresh_state_false_skips_delete_syftbox(tmp_path, monkeypatch):
    """Opting out (fresh_state=False) preserves state across init."""
    monkeypatch.setattr(
        "syft_enclaves.runner.TEE_SOCKET_PATH", tmp_path / "nonexistent"
    )

    client = _make_client()
    runner = EnclaveRunner(client=client, fresh_state=False)
    runner.init()

    client.delete_syftbox.assert_not_called()


def test_fresh_state_default_is_true(tmp_path, monkeypatch):
    """Constructor default unified with settings default — fresh state on by default."""
    monkeypatch.setattr(
        "syft_enclaves.runner.TEE_SOCKET_PATH", tmp_path / "nonexistent"
    )

    client = _make_client()
    runner = EnclaveRunner(client=client)  # no fresh_state arg
    assert runner.fresh_state is True
    runner.init()
    client.delete_syftbox.assert_called_once_with()


def test_fresh_state_uses_default_kwargs_on_delete(tmp_path, monkeypatch):
    """We rely on delete_syftbox's own defaults — no kwargs passed."""
    monkeypatch.setattr(
        "syft_enclaves.runner.TEE_SOCKET_PATH", tmp_path / "nonexistent"
    )

    client = _make_client()
    EnclaveRunner(client=client, fresh_state=True).init()

    # Must be called with no positional or keyword args — let the method's
    # own defaults handle broadcast_delete_events and verbose.
    call = client.delete_syftbox.call_args
    assert call.args == ()
    assert call.kwargs == {}


def _client_with_store(use_encryption: bool):
    client = _make_client()
    store = client._rds.peer_manager.peer_store
    store.use_encryption = use_encryption
    store.public_key.identity_fingerprint.return_value = FAKE_KEY_FINGERPRINT
    return client


def test_publish_attestation_binds_own_key(monkeypatch):
    """The token request carries the fingerprint of the store's identity key."""
    fetch = MagicMock(return_value="signed.jwt")
    monkeypatch.setattr("syft_enclaves.runner.fetch_attestation_token", fetch)
    client = _client_with_store(use_encryption=True)

    EnclaveRunner(client=client, fresh_state=False)._publish_attestation()

    assert fetch.call_args.kwargs["eat_nonce"][1] == FAKE_KEY_FINGERPRINT
    assert client._rds.peer_manager.get_own_version().attestation_token == "signed.jwt"


def test_publish_attestation_without_encryption_sends_version_only(monkeypatch):
    fetch = MagicMock(return_value="signed.jwt")
    monkeypatch.setattr("syft_enclaves.runner.fetch_attestation_token", fetch)
    client = _client_with_store(use_encryption=False)

    EnclaveRunner(client=client, fresh_state=False)._publish_attestation()

    assert len(fetch.call_args.kwargs["eat_nonce"]) == 1
