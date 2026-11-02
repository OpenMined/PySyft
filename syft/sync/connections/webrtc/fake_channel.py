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
        """Join ``a`` and ``b``. Both ends are attached before either side
        resends, as both ends of a real channel exist before it opens."""
        a.attach(b.email, FakeChannel(sender=a.email, receiver=b))
        b.attach(a.email, FakeChannel(sender=b.email, receiver=a))
        a.resend(b.email)
        b.resend(a.email)
