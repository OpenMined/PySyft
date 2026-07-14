import html
import os

from syft_client.sync.peers.peer import Peer, PeerState

# Emoji + friendly label shown per peer state in the summary view.
_STATE_DISPLAY = {
    PeerState.ACCEPTED: ("✅", "connected"),
    PeerState.REQUESTED_BY_ME: ("⏳", "requested_by_us (waiting for approval)"),
    PeerState.REQUESTED_BY_PEER: (
        "📩",
        "requested_by_peer (waiting for your approval)",
    ),
    PeerState.REJECTED: ("❌", "rejected"),
}


class PeerList(list):
    def __init__(self, peers: list[Peer]):
        """
        PeerList is a list specifically for Peer objects.
        Validates that all items are Peer objects and that they are sorted correctly.
        """
        super().__init__(peers)
        # Validate all items are Peer objects
        for item in self:
            if not isinstance(item, Peer):
                raise TypeError(
                    f"All items in PeerList must be Peer objects, but got {type(item)}"
                )
        # Validate sorting (approved before pending)
        self._validate_sorting()

    def _validate_sorting(self):
        """Ensure peers are sorted: accepted, then requested_by_me, then requested_by_peer"""
        order = {
            PeerState.ACCEPTED: 0,
            PeerState.REQUESTED_BY_ME: 1,
            PeerState.REQUESTED_BY_PEER: 2,
        }
        sorted_peers = sorted(self, key=lambda p: order.get(p.state, 3))
        if list(self) != sorted_peers:
            raise ValueError(
                "PeerList must be sorted: accepted first, then requested_by_me, then requested_by_peer"
            )

    def __getitem__(self, index: str | int) -> Peer:  # ty:ignore[invalid-method-override]
        if isinstance(index, int):
            return super().__getitem__(index)
        elif isinstance(index, str):
            try:
                return next(peer for peer in self if peer.email == index)
            except StopIteration:
                raise ValueError(f"Peer with email {index} not found")
        else:
            raise ValueError(f"Invalid index type: {type(index)}")

    def _summary_text(self) -> str:
        """Clean, human-friendly summary of the peers and what to do next."""
        if not self:
            return "👥 You have no peers yet."

        lines = [f"👥 Your peers ({len(self)}):", ""]
        pad = max(len(peer.email) for peer in self)
        connected = []
        has_pending = False
        for peer in self:
            emoji, label = _STATE_DISPLAY.get(peer.state, ("❓", str(peer.state.value)))
            lines.append(f"  {emoji} {peer.email.ljust(pad)}  — {label}")
            if peer.state == PeerState.ACCEPTED:
                connected.append(peer)
            elif peer.state in (PeerState.REQUESTED_BY_ME, PeerState.REQUESTED_BY_PEER):
                has_pending = True

        if connected:
            lines += [
                "",
                "💡 Tip: Once connected, you can access their datasets with:",
                f'   client.datasets.get("dataset_name", datasite="{connected[0].email}")',
            ]
        if has_pending:
            lines += [
                "",
                "⏳ You have pending requests waiting for approval — "
                "follow up with the peer or check back later.",
            ]
        return "\n".join(lines)

    def __str__(self) -> str:
        """Clean summary shown by print() and outside notebooks."""
        return self._summary_text()

    def _repr_html_(self) -> str | None:
        """Used by Jupyter to display the summary; falls back to text elsewhere."""
        if "SYFT_NO_REPR_HTML" in os.environ:
            return None
        return f"<pre>{html.escape(self._summary_text())}</pre>"

    def __repr__(self):
        """Technical repr for debugging / normal REPL."""
        peers = [p for p in self]
        return f"PeerList({peers!r})"
