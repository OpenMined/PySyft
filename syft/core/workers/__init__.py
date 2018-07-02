"""Interfaces for communicating about objects between Clients and Workers"""

from .base import BaseWorker
from .socket import SocketWorker
from .virtual import VirtualWorker

__all__ = ['BaseWorker', 'SocketWorker', 'VirtualWorker']
