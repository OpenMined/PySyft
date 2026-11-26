"""In-memory channels between WebRTC users, for tests."""

from syft.sync.connections.webrtc.peer_connections import (
    OnFrame,
    OnOpen,
    PeerConnections,
)


class FakeNetwork:
    """What ``LocalSignaling`` and the machine's network are for aiortc: who
    listens under which email, and which pairs have an open channel. Shared by
    the ``FakePeerConnections`` of a test; it calls back inside each call."""

    def __init__(self):
        # email -> the on_open of whoever listens under it
        self._peer_on_opens: dict[str, OnOpen] = {}
        # email -> the on_frame of whoever listens under it
        self._peer_on_frames: dict[str, OnFrame] = {}
        # Pairs of emails with an open channel between them.
        self._channels: set[frozenset[str]] = set()

    def listen(self, email: str, on_open: OnOpen, on_frame: OnFrame) -> None:
        self._peer_on_opens[email] = on_open
        self._peer_on_frames[email] = on_frame

    def connect(self, email: str, peer: str) -> None:
        """Open a channel between ``email`` and ``peer``, and tell both."""
        if peer not in self._peer_on_opens:
            raise ConnectionError(f"{peer} is not listening")
        self._channels.add(frozenset((email, peer)))
        self._peer_on_opens[email](peer)
        self._peer_on_opens[peer](email)

    def is_listening(self, email: str) -> bool:
        return email in self._peer_on_opens

    def has_channel(self, email: str, peer: str) -> bool:
        return frozenset((email, peer)) in self._channels

    def send(self, sender: str, receiver: str, frame: bytes) -> None:
        """Hand ``frame`` to ``receiver`` inside this call; dropped when the two
        have no open channel."""
        if frozenset((sender, receiver)) in self._channels:
            self._peer_on_frames[receiver](sender, frame)

    def leave(self, email: str) -> None:
        """Stop listening under ``email`` and close its channels."""
        self._peer_on_opens.pop(email, None)
        self._peer_on_frames.pop(email, None)
        self._channels = {pair for pair in self._channels if email not in pair}


class FakePeerConnections(PeerConnections):
    """``PeerConnections`` over a ``FakeNetwork``, in place of aiortc. Fully
    synchronous: ``send`` hands the frame to the peer inside the call, and
    ``connect`` opens both ends before it returns."""

    def __init__(self, network: FakeNetwork):
        self.network = network
        self.email: str | None = None

    def open(self, email: str, on_open: OnOpen, on_frame: OnFrame) -> None:
        self.email = email
        self.network.listen(email, on_open, on_frame)

    def connect(self, peer: str) -> None:
        self.network.connect(self.email, peer)

    def set_peers(self, peers: list[str]) -> None:
        """Connect to each of ``peers`` that listens and has no channel to this
        client. A fake network has no roles, so either side connects."""
        for peer in peers:
            if self.network.is_listening(peer) and not self.network.has_channel(
                self.email, peer
            ):
                self.network.connect(self.email, peer)

    def maybe_raise_background_errors(self) -> None:
        """Do nothing: the fake network does no work in the background."""

    def send(self, peer: str, frame: bytes) -> None:
        self.network.send(self.email, peer, frame)

    def close(self) -> None:
        self.network.leave(self.email)
