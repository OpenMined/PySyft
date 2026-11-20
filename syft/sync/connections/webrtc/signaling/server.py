"""The signaling server for syft's WebRTC peers.

A connection carries one signaling session or one peering request. In a
session, the server relays offers and answers between two clients that are
online. It tells a client when a watched peer comes online or goes offline, and
it gives a client ICE servers for each new peer connection. A peering request
stores or reads peer requests and the documents that clients publish for their
peers. The server keeps all state in memory.

Each message is one JSON object with a "type" field. A session starts with
``hello {email, access_key}``, and the answer is ``welcome``. Then the client
sends these messages:

- ``watch {peers}``: ``peers`` replaces the earlier watch list of the client.
  The server sends ``online {email}`` for each watched peer
  that is online. Later it sends ``online`` or ``offline`` when a watched peer
  connects or disconnects.
- ``offer {to, id, sdp}``, ``answer {to, id, sdp}`` and ``refuse {to, id,
  reason}``: the peer gets the message with ``from`` in place of ``to``. A
  client refuses an offer from an email that is not its peer.
- ``get_ice_servers {id}``: the answer is ``ice_servers {id, ice_servers}``,
  with new short-lived TURN credentials.

A peering request also has ``email`` and ``access_key``. It is one of these:

- ``request_peer {peer}``: ask ``peer`` to peer. The answer is ``ok``.
- ``list_peer_requests``: the answer is ``peer_requests {emails}``, the emails
  that asked to peer with the client.
- ``put_document {name, data}``: store a document of the client. The answer is
  ``ok``.
- ``grant_document {name, reader}``: let ``reader`` read a document of the
  client. The answer is ``ok``.
- ``get_document {owner, name}``: the answer is ``document {data}``. ``data``
  is null if the document does not exist, or if the client may not read it.

After a protocol error, the server sends ``error {reason}`` and closes the
connection. A message to a peer that is offline gets ``error {id, reason}``,
and the session continues.

Emails are compared in lower case. The server reports each email as its owner
spells it in a hello or a peering request. A second sign-in with the same email
replaces the first session.

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
from typing import TypeVar

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
# (email of the client, its connection, a message of its session)
SessionHandler = Callable[[str, ServerConnection, dict], Awaitable[None]]
# (email of the client, its peering request) -> the answer
PeeringHandler = Callable[[str, dict], dict]
Handler = TypeVar("Handler")
OK_ANSWER = {"type": "ok"}
# relayed message type -> the field that the server copies to the peer
RELAYED_FIELDS = {"offer": "sdp", "answer": "sdp", "refuse": "reason"}
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


@dataclass(frozen=True)
class SignedInClient:
    # The email as the client spells it in its hello.
    email: str
    connection: ServerConnection


class PeeringStore:
    """The peer requests, and the documents that clients publish for peers.

    Emails compare in lower case. A request keeps the email of its sender as
    the sender spells it.
    """

    def __init__(self):
        # email in lower case -> the senders that asked to peer with it, as
        # sender email in lower case -> sender email as the sender spells it
        self._requests: dict[str, dict[str, str]] = defaultdict(dict)
        # (owner email in lower case, document name) -> the document
        self._documents: dict[tuple[str, str], str] = {}
        # (owner email in lower case, document name) -> the emails in lower case
        # that may read it
        self._readers: dict[tuple[str, str], set[str]] = defaultdict(set)

    def add_request(self, sender: str, peer: str) -> None:
        self._requests[peer.lower()][sender.lower()] = sender

    def list_requests(self, email: str) -> list[str]:
        """Return the emails that asked to peer with ``email``."""
        return sorted(self._requests.get(email.lower(), {}).values())

    def put_document(self, owner: str, name: str, data: str) -> None:
        self._documents[(owner.lower(), name)] = data

    def grant_document(self, owner: str, name: str, reader: str) -> None:
        self._readers[(owner.lower(), name)].add(reader.lower())

    def get_document(self, reader: str, owner: str, name: str) -> str | None:
        """Return the document, or None if it does not exist or ``reader`` may
        not read it."""
        key = (owner.lower(), name)
        readers = self._readers.get(key, set())
        if reader.lower() != owner.lower() and reader.lower() not in readers:
            return None
        return self._documents.get(key)


class SignalingServer:
    """Relays signaling messages between the clients that are online, and
    answers peering requests."""

    def __init__(self, config: SignalingServerConfig, ice_servers: IceServerSource):
        self._config = config
        self._ice_servers = ice_servers
        # email in lower case -> the client that signed in with it
        self._clients: dict[str, SignedInClient] = {}
        # email in lower case -> the connections that watch it
        self._watchers: dict[str, set[ServerConnection]] = defaultdict(set)
        self._peering = PeeringStore()
        # message type -> the method that handles it in a session
        self._session_handlers: dict[str, SessionHandler] = {
            "watch": self._watch_peers,
            "offer": self._relay_to_peer,
            "answer": self._relay_to_peer,
            "refuse": self._relay_to_peer,
            "get_ice_servers": self._send_ice_servers,
        }
        # peering request type -> the method that runs it
        self._peering_handlers: dict[str, PeeringHandler] = {
            "request_peer": self._request_peer,
            "list_peer_requests": self._list_peer_requests,
            "put_document": self._put_document,
            "grant_document": self._grant_document,
            "get_document": self._get_document,
        }

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
        try:
            first = read_message(await connection.recv())
            if first["type"] == "hello":
                await self._run_session(connection, first)
            else:
                await self._answer_peering_request(connection, first)
        except ProtocolError as err:
            await connection.send(json.dumps({"type": "error", "reason": str(err)}))
            await connection.close(CloseCode.POLICY_VIOLATION, "protocol error")

    async def _run_session(self, connection: ServerConnection, hello: dict) -> None:
        email = await self._sign_in(connection, hello)
        try:
            async for raw in connection:
                await self._dispatch_message(email, connection, read_message(raw))
        finally:
            self._remove_watcher(connection)
            # A replaced connection is not the current one; its email stays online.
            current = self._clients.get(email.lower())
            if current is not None and current.connection is connection:
                self._sign_out(email)

    async def _sign_in(self, connection: ServerConnection, hello: dict) -> str:
        email = self._authenticate(hello)
        await connection.send(json.dumps({"type": "welcome"}))
        replaced = self._clients.get(email.lower())
        self._clients[email.lower()] = SignedInClient(email, connection)
        logger.info(f"{email} signed in")
        if replaced is not None:
            await replaced.connection.close(REPLACED_CLOSE_CODE, "signed in again")
        self._notify_watchers(email, "online")
        return email

    def _authenticate(self, message: dict) -> str:
        """Check the access key in a hello or a peering request.

        Returns:
            The email of the client, as the client spells it.

        Raises:
            ProtocolError: The access key is wrong.
        """
        access_key = read_text_field(message, "access_key")
        if not hmac.compare_digest(
            access_key.encode(), self._config.access_key.encode()
        ):
            raise ProtocolError("The access key is wrong")
        return read_text_field(message, "email")

    async def _answer_peering_request(
        self, connection: ServerConnection, request: dict
    ) -> None:
        email = self._authenticate(request)
        handler = find_handler(self._peering_handlers, request["type"])
        await connection.send(json.dumps(handler(email, request)))

    def _request_peer(self, email: str, request: dict) -> dict:
        self._peering.add_request(email, read_text_field(request, "peer"))
        return OK_ANSWER

    def _list_peer_requests(self, email: str, request: dict) -> dict:
        return {"type": "peer_requests", "emails": self._peering.list_requests(email)}

    def _put_document(self, email: str, request: dict) -> dict:
        name = read_text_field(request, "name")
        self._peering.put_document(email, name, read_text_field(request, "data"))
        return OK_ANSWER

    def _grant_document(self, email: str, request: dict) -> dict:
        name = read_text_field(request, "name")
        reader = read_text_field(request, "reader")
        self._peering.grant_document(email, name, reader)
        return OK_ANSWER

    def _get_document(self, email: str, request: dict) -> dict:
        owner = read_text_field(request, "owner")
        name = read_text_field(request, "name")
        return {
            "type": "document",
            "data": self._peering.get_document(email, owner, name),
        }

    async def _dispatch_message(
        self, email: str, connection: ServerConnection, message: dict
    ) -> None:
        handler = find_handler(self._session_handlers, message["type"])
        await handler(email, connection, message)

    async def _watch_peers(
        self, email: str, connection: ServerConnection, message: dict
    ) -> None:
        # The new list replaces the earlier one.
        self._remove_watcher(connection)
        for peer in read_text_list_field(message, "peers"):
            self._watchers[peer.lower()].add(connection)
            if peer.lower() in self._clients:
                online = {"type": "online", "email": self._clients[peer.lower()].email}
                await connection.send(json.dumps(online))

    async def _send_ice_servers(
        self, email: str, connection: ServerConnection, message: dict
    ) -> None:
        request_id = read_text_field(message, "id")
        ice_servers = await self._ice_servers()
        reply = {"type": "ice_servers", "id": request_id, "ice_servers": ice_servers}
        await connection.send(json.dumps(reply))

    async def _relay_to_peer(
        self, sender: str, connection: ServerConnection, message: dict
    ) -> None:
        receiver_email = read_text_field(message, "to")
        request_id = read_text_field(message, "id")
        field = RELAYED_FIELDS[message["type"]]
        relayed = json.dumps(
            {
                "type": message["type"],
                "from": sender,
                "id": request_id,
                field: read_text_field(message, field),
            }
        )
        receiver = self._clients.get(receiver_email.lower())
        if receiver is None:
            await send_offline_error(connection, request_id, receiver_email)
            return
        try:
            await receiver.connection.send(relayed)
        except ConnectionClosed:
            # The receiver left before its own handler signed it out.
            await send_offline_error(connection, request_id, receiver_email)

    def _notify_watchers(self, email: str, kind: str) -> None:
        """Send ``kind`` (online or offline) for ``email`` to its watchers."""
        event = json.dumps({"type": kind, "email": email})
        # broadcast skips a watcher whose connection is closed.
        broadcast(self._watchers[email.lower()], event)

    def _remove_watcher(self, connection: ServerConnection) -> None:
        for watchers in self._watchers.values():
            watchers.discard(connection)

    def _sign_out(self, email: str) -> None:
        del self._clients[email.lower()]
        logger.info(f"{email} signed out")
        self._notify_watchers(email, "offline")


def find_handler(handlers: dict[str, Handler], kind: str) -> Handler:
    """Return the handler for a message of type ``kind``.

    Raises:
        ProtocolError: No handler takes ``kind``.
    """
    if kind not in handlers:
        expected = ", ".join(handlers)
        raise ProtocolError(f"Expected a message of type {expected}, got {kind!r}")
    return handlers[kind]


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
