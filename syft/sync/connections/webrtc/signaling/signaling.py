"""How two WebRTC peers swap session descriptions before they connect."""

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Any

# (sender email, offer sdp) -> answer sdp
OnOffer = Callable[[str, str], Awaitable[str]]
# The signaling calls it with the email of a watched peer that is online.
OnOnline = Callable[[str], None]
# Each item is an RTCIceServer as JSON: "urls", and "username" and
# "credential" for a TURN server.
IceServers = list[dict[str, Any]]


class Signaling(ABC):
    """Carries one peer's offer to another and the answer back."""

    @abstractmethod
    async def listen(
        self, email: str, on_offer: OnOffer, on_online: OnOnline
    ) -> IceServers:
        """Listen for offers and online peers as ``email``.

        ``on_offer`` answers each offer to ``email``. ``on_online`` gets the
        email of each watched peer that is online. Both callbacks run on the
        loop that called ``listen``.

        Returns:
            The ICE servers for the peer connections of ``email``.
        """

    @abstractmethod
    async def watch(self, email: str, peers: list[str]) -> None:
        """Call the ``on_online`` of ``email`` when one of ``peers`` is online.

        If a peer is online now, the first call comes at once. The signaling
        calls ``on_online`` again each time the peer comes online. ``peers``
        replaces the peers of an earlier call.
        """

    @abstractmethod
    async def offer(self, sender: str, receiver: str, sdp: str) -> str:
        """Send ``sender``'s offer to ``receiver``; return its answer."""

    @abstractmethod
    async def close(self, email: str) -> None:
        """Close the signaling for ``email``.

        Then ``email`` gets no more offers, and its ``on_online`` gets no more
        calls.
        """
