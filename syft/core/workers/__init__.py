"""Interfaces for communicating about objects between Clients and Workers."""

from syft.core.workers.base import BaseWorker
from syft.core.workers.socket import SocketWorker
from syft.core.workers.virtual import VirtualWorker
from syft.core.workers.websocket import WebSocketWorker

__all__ = ["BaseWorker", "SocketWorker", "VirtualWorker", "WebSocketWorker"]
