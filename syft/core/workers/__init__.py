"""Interfaces for communicating about objects between Clients and Workers"""

from .base import BaseWorker
from .socket import SocketWorker
from .virtual import VirtualWorker
from .websocket import WebSocketWorker

__all__ = ['BaseWorker', 'SocketWorker', 'VirtualWorker', 'WebSocketWorker']
