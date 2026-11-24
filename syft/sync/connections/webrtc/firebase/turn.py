"""ICE servers from the TURN Cloud Function of a Firebase project."""

import asyncio

from syft.sync.connections.webrtc.firebase.auth import FirebaseSession
from syft.sync.connections.webrtc.firebase.rest import post_json
from syft.sync.connections.webrtc.signaling.signaling import IceServerSet


class TurnFunction:
    """An ``IceServerSource``: each call gets new short-lived TURN credentials
    from the function in firebase/functions/main.py."""

    def __init__(self, url: str, session: FirebaseSession):
        self._url = url
        self._session = session

    async def __call__(self) -> IceServerSet:
        """Raises:
        ConnectionError: The function refuses the user.
        ConnectionUnavailableError: The function cannot be reached.
        """
        return await asyncio.to_thread(self._request_ice_servers)

    def _request_ice_servers(self) -> IceServerSet:
        reply = post_json(self._url, {}, self._session.id_token())
        return IceServerSet(reply["ice_servers"], reply["lifetime_seconds"])


async def no_ice_servers() -> IceServerSet:
    """An ``IceServerSource`` for peers on one machine, which reach each other
    with no STUN or TURN server."""
    return IceServerSet([], None)
