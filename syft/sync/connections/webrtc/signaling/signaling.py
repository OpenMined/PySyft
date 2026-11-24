"""How two WebRTC peers find each other online, and swap session descriptions
before they connect."""

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

# (sender email, offer sdp) -> answer sdp
OnOffer = Callable[[str, str], Awaitable[str]]
# (sender email, sender session) for each hello. A session is a new id at each
# sign-in of the sender.
OnHello = Callable[[str, str], None]
# The signaling calls it with the reason when a session ends before ``close``.
OnClosed = Callable[[ConnectionError], None]
# The signaling calls it with the reason when it fails to answer an offer for a
# reason other than a refusal.
OnAnswerFailed = Callable[[ConnectionError], None]
# Each item is an RTCIceServer as JSON: "urls", and "username" and
# "credential" for a TURN server.
IceServers = list[dict[str, Any]]


@dataclass(frozen=True)
class IceServerSet:
    servers: IceServers
    # How long the TURN credentials work, or None if they do not expire.
    lifetime_seconds: float | None


# Each call returns the ICE servers for one new peer connection.
IceServerSource = Callable[[], Awaitable[IceServerSet]]


class Signaling(ABC):
    """Tells peers that a client is online, and carries one peer's offer to
    another and the answer back."""

    @abstractmethod
    async def listen(
        self,
        email: str,
        on_offer: OnOffer,
        on_hello: OnHello,
        on_closed: OnClosed,
        on_answer_failed: OnAnswerFailed,
    ) -> None:
        """Listen for hellos and offers to ``email``.

        ``on_offer`` answers each offer. ``on_hello`` gets each hello.
        ``on_closed`` gets the reason when the session ends before ``close``.
        ``on_answer_failed`` gets the reason when an answer fails for another
        reason than a refusal. The callbacks run on the loop that called
        ``listen``.
        """

    @abstractmethod
    async def ice_servers(self, email: str) -> IceServerSet:
        """Return the ICE servers for a new peer connection of ``email``."""

    @abstractmethod
    async def hello(self, email: str, peers: list[str]) -> None:
        """Tell each of ``peers`` that ``email`` is online.

        A peer that is offline never gets the hello.
        """

    @abstractmethod
    async def offer(self, sender: str, receiver: str, sdp: str) -> str:
        """Send ``sender``'s offer to ``receiver``; return its answer."""

    @abstractmethod
    async def close(self, email: str) -> None:
        """Close the signaling for ``email``.

        Then ``email`` gets no more hellos and no more offers.
        """
