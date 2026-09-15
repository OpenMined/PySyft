"""Tests for EnclaveRunner — fresh_state init behavior and the attest phase."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from syft_enclaves.attestation.envelope import (
    AttestationEvidence,
    AttestationKind,
    tinfoil_evidence,
)
from syft_enclaves.runner import EnclaveRunner

TINFOIL_DOC = {
    "format": "https://tinfoil.sh/predicate/sev-snp-guest/v2",
    "body": "H4sIAAAAAAAA/2JmgAEEixBg",
}


def _make_client():
    """Build a stub SyftEnclaveClient just deep enough for the init phases."""
    client = MagicMock()
    client.email = "enclave@openmined.org"
    client.syftbox_folder = "/tmp/SyftBox_enclave"
    # _on_peering needs these
    client.peers = []
    return client


def test_fresh_state_true_invokes_delete_syftbox():
    """With fresh_state=True (default), _on_initializing must wipe state once."""
    client = _make_client()
    runner = EnclaveRunner(client=client, fresh_state=True, attestation_provider="none")
    runner.init()

    client.delete_syftbox.assert_called_once_with()


def test_fresh_state_false_skips_delete_syftbox():
    """Opting out (fresh_state=False) preserves state across init."""
    client = _make_client()
    runner = EnclaveRunner(
        client=client, fresh_state=False, attestation_provider="none"
    )
    runner.init()

    client.delete_syftbox.assert_not_called()


def test_fresh_state_default_is_true():
    """Constructor default unified with settings default — fresh state on by default."""
    client = _make_client()
    runner = EnclaveRunner(client=client, attestation_provider="none")
    assert runner.fresh_state is True
    runner.init()
    client.delete_syftbox.assert_called_once_with()


def test_fresh_state_uses_default_kwargs_on_delete():
    """We rely on delete_syftbox's own defaults — no kwargs passed."""
    client = _make_client()
    EnclaveRunner(client=client, fresh_state=True, attestation_provider="none").init()

    # Must be called with no positional or keyword args — let the method's
    # own defaults handle broadcast_delete_events and verbose.
    call = client.delete_syftbox.call_args
    assert call.args == ()
    assert call.kwargs == {}


class TestAttestPhase:
    def test_no_tee_and_require_tee_names_every_probed_path(self):
        runner = EnclaveRunner(
            client=_make_client(), require_tee=True, attestation_provider="none"
        )
        # "none" is not a TEE, so require_tee must refuse to start and say
        # where it looked — an operator has two targets to check.
        with pytest.raises(RuntimeError, match="No TEE detected") as excinfo:
            runner.init()
        message = str(excinfo.value)
        assert "/run/container_launcher/teeserver.sock" in message
        assert "/tinfoil/attestation.json" in message

    def test_no_tee_without_require_tee_publishes_nothing(self):
        client = _make_client()
        EnclaveRunner(client=client, attestation_provider="none").init()
        client._rds.peer_manager.write_own_version.assert_not_called()

    def test_publishes_evidence_to_the_version_file(self, monkeypatch):
        evidence = tinfoil_evidence(TINFOIL_DOC, repo="OpenMined/x")
        provider = MagicMock()
        provider.kind = AttestationKind.TINFOIL
        provider.collect.return_value = evidence
        monkeypatch.setattr(
            "syft_enclaves.runner.select_provider", lambda name, settings: provider
        )

        client = _make_client()
        version = MagicMock(extra={})
        client._rds.peer_manager.get_own_version.return_value = version

        EnclaveRunner(client=client, require_tee=True).init()

        assert AttestationEvidence.read_from(version) == evidence
        client._rds.peer_manager.write_own_version.assert_called_once_with()

    def test_the_key_bundle_is_published_for_the_attestation_endpoint(
        self, monkeypatch
    ):
        """Regression: the publish call was defined but never wired in.

        Without it the /attestation endpoint serves no key bundle, so a peer
        gets a pinned channel with nothing bound to it.
        """
        provider = MagicMock()
        provider.kind = AttestationKind.TINFOIL
        provider.collect.return_value = tinfoil_evidence(TINFOIL_DOC)
        monkeypatch.setattr(
            "syft_enclaves.runner.select_provider", lambda name, settings: provider
        )
        written = {}
        monkeypatch.setattr(
            "syft_enclaves.runner.write_public_bundle",
            lambda bundle, keys_path: written.update(
                bundle=bundle, keys_path=keys_path
            ),
        )

        client = _make_client()
        client._rds.peer_manager.syftbox_folder = Path("/tmp/SyftBox_enclave")
        store = client._rds.peer_manager.peer_store
        store.email = "enclave@openmined.org"
        store.use_encryption = True
        store.has_my_keys.return_value = True
        store.get_public_bundle.return_value = {"identity": "enclave@openmined.org"}

        EnclaveRunner(client=client, require_tee=True).init()

        assert written["bundle"] == {"identity": "enclave@openmined.org"}
        # The endpoint needs the private key location to answer a nonce.
        assert written["keys_path"].name == "crypto_keys.json"
        # And the keys must be back on disk: fresh_state wiped the folder
        # moments earlier, so the file the endpoint signs with is gone.
        store.save_keys.assert_called_once_with(written["keys_path"])

    def test_no_key_bundle_is_published_without_encryption(self, monkeypatch):
        provider = MagicMock()
        provider.kind = AttestationKind.TINFOIL
        provider.collect.return_value = tinfoil_evidence(TINFOIL_DOC)
        monkeypatch.setattr(
            "syft_enclaves.runner.select_provider", lambda name, settings: provider
        )
        calls = []
        monkeypatch.setattr(
            "syft_enclaves.runner.write_public_bundle",
            lambda bundle, keys_path: calls.append(bundle),
        )

        client = _make_client()
        client._rds.peer_manager.syftbox_folder = Path("/tmp/SyftBox_enclave")
        client._rds.peer_manager.peer_store.use_encryption = False

        EnclaveRunner(client=client, require_tee=True).init()

        assert calls == []

    def test_provider_gets_the_settings_object(self, monkeypatch):
        seen = {}

        def fake_select(name, settings):
            seen["name"], seen["settings"] = name, settings
            return None

        monkeypatch.setattr("syft_enclaves.runner.select_provider", fake_select)
        settings = object()
        EnclaveRunner(
            client=_make_client(), attestation_provider="tinfoil", settings=settings
        ).init()

        assert seen == {"name": "tinfoil", "settings": settings}
