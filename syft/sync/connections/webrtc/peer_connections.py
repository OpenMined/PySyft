"""What carries frames between one user and their peers over WebRTC."""

from abc import ABC, abstractmethod
from collections.abc import Callable

# Called with the peer's email when the channel to it opens.
OnOpen = Callable[[str], None]
# Called with the peer's email and each frame that arrives from it.
OnFrame = Callable[[str, bytes], None]


class PeerConnections(ABC):
    """One ordered, reliable channel per peer. Real WebRTC in
    ``AiortcPeerConnections``, in memory in ``FakePeerConnections``."""

    @abstractmethod
    def open(self, email: str, on_open: OnOpen, on_frame: OnFrame) -> None:
        """Accept peers as ``email``. ``on_open`` and ``on_frame`` report
        each channel that opens and each frame that arrives."""

    @abstractmethod
    def connect(self, peer: str) -> None:
        """Open a channel to ``peer``; return once it is open."""

    @abstractmethod
    def send(self, peer: str, frame: bytes) -> None:
        """Send ``frame`` to ``peer``; dropped when there is no channel to it."""

    @abstractmethod
    def close(self) -> None:
        """Close every channel."""
