"""An in-memory channel between two WebRTC cores, for tests."""

from syft.sync.connections.webrtc.channel import Channel
from syft.sync.connections.webrtc.webrtc_p2p import WebRTCP2PConnection


class FakeChannel(Channel):
    """One end of an in-memory pipe, in place of an aiortc data channel. A
    frame reaches the other core inside ``send``: no threads, no network."""

    def __init__(self, sender: str, receiver: WebRTCP2PConnection):
        self.sender = sender
        self.receiver = receiver

    def send(self, frame: bytes) -> None:
        self.receiver.on_frame(self.sender, frame)

    @staticmethod
    def connect(a: WebRTCP2PConnection, b: WebRTCP2PConnection) -> None:
        """Join ``a`` and ``b`` and open the channel. Both ends are attached
        before it opens, as with a real channel, so the first receipts have
        a way back."""
        a.attach(b.email, FakeChannel(sender=a.email, receiver=b))
        b.attach(a.email, FakeChannel(sender=b.email, receiver=a))
        a.on_open(b.email)
        b.on_open(a.email)
