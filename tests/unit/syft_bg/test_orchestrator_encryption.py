"""The syft-bg orchestrators build their client with the configured policy."""

from unittest.mock import MagicMock

import pytest
import syft_rds
from syft_rds import EncryptionPolicy

from syft_bg.approve.config import AutoApproveConfig
from syft_bg.approve.orchestrator import ApprovalOrchestrator
from syft_bg.sync.config import SyncConfig
from syft_bg.sync.orchestrator import SyncOrchestrator


def _sync(temp_dir, policy):
    config = SyncConfig(
        do_email="do@test.com",
        syftbox_root=temp_dir,
        encryption_policy=policy,
        sync_state_path=temp_dir / "sync_state.json",
    )
    return SyncOrchestrator, config


def _approve(temp_dir, policy):
    config = AutoApproveConfig(
        do_email="do@test.com", syftbox_root=temp_dir, encryption_policy=policy
    )
    return ApprovalOrchestrator, config


@pytest.fixture
def client_kwargs(monkeypatch):
    """Capture the keyword arguments the orchestrator builds its client with."""
    seen = {}
    monkeypatch.setattr(syft_rds, "check_env", lambda: syft_rds.Environment.JUPYTER)
    monkeypatch.setattr(
        syft_rds.SyftRDSClientConfig,
        "for_jupyter",
        classmethod(lambda cls, **kwargs: seen.update(kwargs)),
    )
    monkeypatch.setattr(
        syft_rds.SyftRDSClient,
        "from_config",
        classmethod(lambda cls, config: MagicMock()),
    )
    monkeypatch.setattr(
        ApprovalOrchestrator, "_wait_for_sync_ready", staticmethod(lambda **kw: None)
    )
    return seen


@pytest.mark.parametrize("build", [_sync, _approve], ids=["sync", "approve"])
def test_orchestrator_uses_configured_policy(temp_dir, client_kwargs, build):
    orchestrator, config = build(temp_dir, EncryptionPolicy.REQUIRED)
    orchestrator.from_config(config)
    assert client_kwargs["encryption"] == EncryptionPolicy.REQUIRED


@pytest.mark.parametrize("build", [_sync, _approve], ids=["sync", "approve"])
def test_orchestrator_refuses_missing_policy(temp_dir, client_kwargs, build):
    orchestrator, config = build(temp_dir, None)
    with pytest.raises(ValueError, match="encryption_policy"):
        orchestrator.from_config(config)
