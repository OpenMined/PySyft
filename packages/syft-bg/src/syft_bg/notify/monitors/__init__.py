"""Monitors for detecting events to notify about."""

from syft_bg.notify.monitors.job import JobMonitor
from syft_bg.notify.monitors.peer import PeerMonitor

__all__ = ["JobMonitor", "PeerMonitor"]
