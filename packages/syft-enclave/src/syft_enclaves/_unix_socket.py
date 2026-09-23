"""Unix-domain-socket HTTP client.

Shared by the two places that talk to the Confidential Space launcher: the
attestation-evidence provider and the token bootstrap. Deliberately stdlib-only
— ``bootstrap`` runs as its own process before the runner starts, so anything
imported here is paid on every boot.
"""

from __future__ import annotations

import socket
from http.client import HTTPConnection


class UnixSocketConnection(HTTPConnection):
    """``HTTPConnection`` that connects over a Unix domain socket."""

    def __init__(self, socket_path: str):
        super().__init__("localhost")
        self._socket_path = socket_path

    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(self._socket_path)
