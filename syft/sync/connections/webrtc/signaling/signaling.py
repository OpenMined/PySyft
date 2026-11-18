"""How two WebRTC peers swap session descriptions before they connect."""

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Any

# (sender email, offer sdp) -> answer sdp
OnOffer = Callable[[str, str], Awaitable[str]]
# Each item is an RTCIceServer as JSON: "urls", and "username" and
# "credential" for a TURN server.
IceServers = list[dict[str, Any]]


class Signaling(ABC):
    """Carries one peer's offer to another and the answer back."""

    @abstractmethod
    async def listen(self, email: str, on_offer: OnOffer) -> None:
        """Answer offers for ``email`` with ``on_offer``, run on the loop that
        called ``listen``."""

    @abstractmethod
    async def offer(self, sender: str, receiver: str, sdp: str) -> str:
        """Send ``sender``'s offer to ``receiver``; return its answer."""
