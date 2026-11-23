"""How two WebRTC peers swap session descriptions before they connect."""

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Any

# (sender email, offer sdp) -> answer sdp
OnOffer = Callable[[str, str], Awaitable[str]]
# The signaling calls it with the email of a watched peer that is online.
OnOnline = Callable[[str], None]
# The signaling calls it with the reason when a session ends before ``close``.
OnClosed = Callable[[ConnectionError], None]
# The signaling calls it with the reason when it fails to answer an offer for a
# reason other than a refusal.
OnAnswerFailed = Callable[[ConnectionError], None]
# Each item is an RTCIceServer as JSON: "urls", and "username" and
# "credential" for a TURN server.
IceServers = list[dict[str, Any]]
# Each call returns the ICE servers for one new peer connection.
IceServerSource = Callable[[], Awaitable[IceServers]]


class Signaling(ABC):
    """Carries one peer's offer to another and the answer back."""

    @abstractmethod
    async def listen(
        self,
        email: str,
        on_offer: OnOffer,
        on_online: OnOnline,
        on_closed: OnClosed,
        on_answer_failed: OnAnswerFailed,
    ) -> None:
        """Listen for offers and online peers as ``email``.

        ``on_offer`` answers each offer to ``email``. ``on_online`` gets the
        email of each watched peer that is online. ``on_closed`` gets the reason
        when the session ends before ``close``. ``on_answer_failed`` gets the
        reason when an answer fails for another reason than a refusal. The
        callbacks run on the loop that called ``listen``.
        """

    @abstractmethod
    async def ice_servers(self, email: str) -> IceServers:
        """Return the ICE servers for a new peer connection of ``email``."""

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
