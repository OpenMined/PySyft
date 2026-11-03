"""The pipe under the WebRTC core: one per peer, carrying frames as bytes."""

from abc import ABC, abstractmethod


class Channel(ABC):
    """One ordered, reliable pipe to one peer. An aiortc data channel in
    production, a ``FakeChannel`` in tests. It hands each frame that arrives
    to the core's ``on_frame``."""

    @abstractmethod
    def send(self, frame: bytes) -> None:
        """Send ``frame`` to the peer."""
