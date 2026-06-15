"""Approval service for auto-approving jobs and peers."""

from syft_bg.approve.config import (
    AutoApproveConfig,
    AutoApprovalObj,
    AutoApprovalsConfig,
    PeerApprovalConfig,
    FileEntry,
    ScriptEntry,
)

__all__ = [
    "AutoApproveConfig",
    "AutoApprovalsConfig",
    "AutoApprovalObj",
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
