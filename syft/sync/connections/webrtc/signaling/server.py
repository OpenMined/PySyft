"""The signaling server for syft's WebRTC peers.

Each client keeps one WebSocket open. The server relays offers and answers
between two clients that are online. It tells a client when a watched peer
comes online or goes offline, and it gives each client its ICE servers. It
keeps all state in memory.

Each message is one JSON object with a "type" field:

- ``hello {email, access_key}``: the first message of a client. The answer is
  ``welcome {ice_servers}``.
- ``watch {peers}``: the server sends ``online {email}`` for each watched peer
  that is online. Later it sends ``online`` or ``offline`` when a watched peer
  connects or disconnects.
- ``offer {to, id, sdp}`` and ``answer {to, id, sdp}``: the peer gets the
  message with ``from`` in place of ``to``.
- ``error {reason}``, with the ``id`` of the request when there is one. After
  a protocol error, the server closes the connection.

Emails are compared in lower case. A second sign-in with the same email
replaces the first connection.

Run it with ``python -m syft.sync.connections.webrtc.signaling.server``. It
reads its settings from these environment variables: ``SYFT_SIGNALING_HOST``,
``SYFT_SIGNALING_PORT``, ``SYFT_SIGNALING_ACCESS_KEY``,
``CLOUDFLARE_TURN_KEY_ID`` and ``CLOUDFLARE_TURN_KEY_API_TOKEN``. It needs the
``webrtc`` extra.
"""

import asyncio
import contextlib
import hmac
import json
import logging
import os
from collections import defaultdict
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass

from websockets.asyncio.server import ServerConnection, broadcast, serve
from websockets.exceptions import ConnectionClosed
from websockets.frames import CloseCode

from syft.sync.connections.webrtc.signaling.cloudflare_turn import (
    CloudflareIceServers,
    CloudflareTurnConfig,
)
from syft.sync.connections.webrtc.signaling.signaling import IceServers

logger = logging.getLogger(__name__)

IceServerSource = Callable[[], Awaitable[IceServers]]
# A client that stays signed in longer than this loses its TURN relay.
CREDENTIALS_TTL_SECONDS = 24 * 3600

# The close code for a connection that a newer sign-in replaced.
REPLACED_CLOSE_CODE = 4000
MAX_PORT = 65535
# An error message quotes at most this many characters of a bad message.
MAX_QUOTED_CHARS = 100


class ProtocolError(Exception):
    """A client sent a message that breaks the protocol."""


@dataclass(frozen=True)
class SignalingServerConfig:
    host: str
    # 0 lets the operating system pick a free port.
    port: int
    # Each client must send this key in its hello.
    access_key: str

    def __post_init__(self) -> None:
        if not 0 <= self.port <= MAX_PORT:
            raise ValueError(f"port must be 0 to {MAX_PORT}, got {self.port}")
        if not self.access_key:
            raise ValueError("access_key must not be empty")


class SignalingServer:
    """Relays signaling messages between the clients that are online."""

    def __init__(self, config: SignalingServerConfig, ice_servers: IceServerSource):
        self._config = config
        self._ice_servers = ice_servers
        # email -> the connection of the client that signed in with it
        self._clients: dict[str, ServerConnection] = {}
        # email -> the connections that watch it
        self._watchers: dict[str, set[ServerConnection]] = defaultdict(set)

    @contextlib.asynccontextmanager
    async def running(self) -> AsyncIterator[int]:
        """Serve until the context ends.

        Yields:
            The port of the first socket that the server listens on.
        """
        async with serve(
            self._handle_connection, self._config.host, self._config.port
        ) as server:
            yield next(iter(server.sockets)).getsockname()[1]

    async def _handle_connection(self, connection: ServerConnection) -> None:
        email = None
        try:
            email = await self._sign_in(connection)
            async for raw in connection:
                await self._dispatch_message(email, connection, read_message(raw))
        except ProtocolError as err:
            await connection.send(json.dumps({"type": "error", "reason": str(err)}))
            await connection.close(CloseCode.POLICY_VIOLATION, "protocol error")
        finally:
            self._remove_watcher(connection)
            # A replaced connection is not the current one; its email stays online.
            if email is not None and self._clients.get(email) is connection:
                self._sign_out(email)

    async def _sign_in(self, connection: ServerConnection) -> str:
        email = await self._read_hello(connection)
        ice_servers = await self._ice_servers()
        await connection.send(
            json.dumps({"type": "welcome", "ice_servers": ice_servers})
        )
        replaced = self._clients.get(email)
        self._clients[email] = connection
        logger.info(f"{email} signed in")
        if replaced is not None:
            await replaced.close(REPLACED_CLOSE_CODE, "signed in again")
        self._notify_watchers(email, "online")
        return email

    async def _read_hello(self, connection: ServerConnection) -> str:
        """Read the hello of a client and check its access key.

        Returns:
            The email of the client, in lower case.

        Raises:
            ProtocolError: The first message is not a hello, or its access key
                is wrong.
        """
        hello = read_message(await connection.recv())
        require_message_type(hello, "hello")
        access_key = read_text_field(hello, "access_key")
        if not hmac.compare_digest(
            access_key.encode(), self._config.access_key.encode()
        ):
            raise ProtocolError("The access key is wrong")
        return read_text_field(hello, "email").lower()

    async def _dispatch_message(
        self, email: str, connection: ServerConnection, message: dict
    ) -> None:
        kind = message["type"]
        if kind == "watch":
            await self._watch_peers(connection, message)
        elif kind in ("offer", "answer"):
            await self._relay_offer_or_answer(email, connection, message)
        else:
            raise ProtocolError(
                f"Expected a watch, offer or answer message, got {kind!r}"
            )

    async def _watch_peers(self, connection: ServerConnection, message: dict) -> None:
        for peer in (p.lower() for p in read_text_list_field(message, "peers")):
            self._watchers[peer].add(connection)
            if peer in self._clients:
                await connection.send(json.dumps({"type": "online", "email": peer}))

    async def _relay_offer_or_answer(
        self, sender: str, connection: ServerConnection, message: dict
    ) -> None:
        receiver_email = read_text_field(message, "to").lower()
        request_id = read_text_field(message, "id")
        relayed = json.dumps(
            {
                "type": message["type"],
                "from": sender,
                "id": request_id,
                "sdp": read_text_field(message, "sdp"),
            }
        )
        receiver = self._clients.get(receiver_email)
        if receiver is None:
            await send_offline_error(connection, request_id, receiver_email)
            return
        try:
            await receiver.send(relayed)
        except ConnectionClosed:
            # The receiver left before its own handler signed it out.
            await send_offline_error(connection, request_id, receiver_email)

    def _notify_watchers(self, email: str, kind: str) -> None:
        """Send ``kind`` (online or offline) for ``email`` to its watchers."""
        event = json.dumps({"type": kind, "email": email})
        # broadcast skips a watcher whose connection is closed.
        broadcast(self._watchers[email], event)

    def _remove_watcher(self, connection: ServerConnection) -> None:
        for watchers in self._watchers.values():
            watchers.discard(connection)

    def _sign_out(self, email: str) -> None:
        del self._clients[email]
        logger.info(f"{email} signed out")
        self._notify_watchers(email, "offline")


def read_message(raw: str | bytes) -> dict:
    """Read the JSON object in one WebSocket message.

    Raises:
        ProtocolError: ``raw`` is binary, is not JSON, or is not an object with
            a text "type" field.
    """
    if not isinstance(raw, str):
        raise ProtocolError("Expected a text message, got a binary one")
    try:
        message = json.loads(raw)
    except json.JSONDecodeError as err:
        raise ProtocolError(f"Expected JSON, got {raw[:MAX_QUOTED_CHARS]!r}") from err
    if not (
        isinstance(message, dict)
        and "type" in message
        and isinstance(message["type"], str)
    ):
        raise ProtocolError(
            f"Expected an object with a type, got {raw[:MAX_QUOTED_CHARS]!r}"
        )
    return message


def require_message_type(message: dict, kind: str) -> None:
    """Check that ``message`` is a ``kind`` message.

    Raises:
        ProtocolError: ``message`` has a different type.
    """
    if message["type"] != kind:
        raise ProtocolError(f"Expected a {kind} message, got {message['type']!r}")


def read_text_field(message: dict, name: str) -> str:
    """Read the text in field ``name`` of ``message``.

    Raises:
        ProtocolError: The field is missing, is empty, or is not text.
    """
    value = message[name] if name in message else None
    if not isinstance(value, str) or not value:
        raise ProtocolError(f"Expected text in {name!r}, got {value!r}")
    return value


def read_text_list_field(message: dict, name: str) -> list[str]:
    """Read the list of texts in field ``name`` of ``message``.

    Raises:
        ProtocolError: The field is missing, or is not a list of texts.
    """
    value = message[name] if name in message else None
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ProtocolError(f"Expected a list of texts in {name!r}, got {value!r}")
    return value


async def send_offline_error(
    connection: ServerConnection, request_id: str, email: str
) -> None:
    """Tell the sender of request ``request_id`` that ``email`` is offline."""
    error = {"type": "error", "id": request_id, "reason": f"{email} is offline"}
    await connection.send(json.dumps(error))


async def serve_forever(server: SignalingServer) -> None:
    """Run ``server`` until the process stops."""
    async with server.running() as port:
        logger.info(f"listening on port {port}")
        await asyncio.Future()


def main() -> None:
    """Run the server with the settings in the environment variables."""
    logging.basicConfig(level=logging.INFO)
    server_config = SignalingServerConfig(
        host=os.environ["SYFT_SIGNALING_HOST"],
        port=int(os.environ["SYFT_SIGNALING_PORT"]),
        access_key=os.environ["SYFT_SIGNALING_ACCESS_KEY"],
    )
    turn_config = CloudflareTurnConfig(
        key_id=os.environ["CLOUDFLARE_TURN_KEY_ID"],
        api_token=os.environ["CLOUDFLARE_TURN_KEY_API_TOKEN"],
        ttl_seconds=CREDENTIALS_TTL_SECONDS,
    )
    server = SignalingServer(server_config, CloudflareIceServers(turn_config))
    asyncio.run(serve_forever(server))


if __name__ == "__main__":
    main()
