"""Approval service: runs api calls, and approves peers."""

from syft_bg.approve.config import (
    AutoApproveConfig,
    AutoApprovalsConfig,
    PeerApprovalConfig,
    FileEntry,
    ScriptEntry,
)

__all__ = [
    "AutoApproveConfig",
    "AutoApprovalsConfig",
    "FileEntry",
    "ScriptEntry",
    "PeerApprovalConfig",
    "ApprovalOrchestrator",
]


def __getattr__(name: str):
    if name == "ApprovalOrchestrator":
        from syft_bg.approve.orchestrator import ApprovalOrchestrator

        return ApprovalOrchestrator
    raise AttributeError(f"module 'syft_bg.approve' has no attribute {name!r}")
