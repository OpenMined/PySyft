from syft_client.sync.peers.peer import Peer, PeerState
from syft_client.sync.reprs.peer_repr import get_peer_list_table


class PeerList(list):
    def __init__(self, *args: Peer, **kwargs):
        """
        PeerList is a list specifically for Peer objects.
        Validates that all items are Peer objects and that they are sorted correctly.
        """
        super().__init__(*args, **kwargs)
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
        last_order = -1
        for peer in self:
            peer_order = order.get(peer.state, 3)
            if peer_order < last_order:
                raise ValueError(
                    "PeerList must be sorted: accepted first, then requested_by_me, then requested_by_peer"
                )
            last_order = peer_order

    def __getitem__(self, index: str | int) -> Peer:
        if isinstance(index, int):
            return super().__getitem__(index)
        elif isinstance(index, str):
            key = index
            for peer in self:
                if peer.email == key:
                    return peer
            raise ValueError(f"Peer with email {index} not found")
        else:
            raise ValueError(f"Invalid index type: {type(index)}")

    def _repr_html_(self) -> str:
        """Used by Jupyter to display Rich HTML."""
        peers = [p for p in self]
        return get_peer_list_table(peers)

    def __repr__(self):
        """Fallback for normal REPL"""
        peers = [p for p in self]
        return f"PeerList({peers!r})"
