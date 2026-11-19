"""Signaling through the syft signaling server in ``server.py``, over one
WebSocket."""

import asyncio
import json
import logging
import uuid
from collections.abc import Coroutine
from dataclasses import dataclass
from typing import Any

from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosedError

from syft.sync.connections.webrtc.signaling.signaling import (
    IceServers,
    OnOffer,
    OnOnline,
    Signaling,
)

logger = logging.getLogger(__name__)

WEBSOCKET_SCHEMES = ("ws://", "wss://")


@dataclass(frozen=True)
class ServerSignalingConfig:
    # The WebSocket URL of the signaling server.
    url: str
    # The shared access key of the signaling server.
    access_key: str

    def __post_init__(self) -> None:
        if not self.url.startswith(WEBSOCKET_SCHEMES):
            raise ValueError(f"url must start with ws:// or wss://, got {self.url!r}")
        if not self.access_key:
            raise ValueError("access_key must not be empty")


class ServerSignaling(Signaling):
    """``Signaling`` for one email, through the signaling server.

    ``listen`` signs in to the server and keeps the WebSocket open until
    ``close``.
    """

    # listen() sets these.
    _connection: ClientConnection
    _reader: asyncio.Task
    _on_offer: OnOffer
    _on_online: OnOnline

    def __init__(self, config: ServerSignalingConfig):
        self._config = config
        # request id -> the answer to an offer of this client
        self._pending_answers: dict[str, asyncio.Future[str]] = {}
        # The tasks that answer offers, until they end. This set keeps a strong
        # reference to each task, because asyncio keeps only a weak one.
        self._answer_tasks: set[asyncio.Task] = set()

    async def listen(
        self, email: str, on_offer: OnOffer, on_online: OnOnline
    ) -> IceServers:
        """Sign in to the server as ``email``.

        Raises:
            ConnectionError: The server refuses the sign-in.
        """
        self._on_offer = on_offer
        self._on_online = on_online
        connection = await connect(self._config.url)
        hello = {"type": "hello", "email": email, "access_key": self._config.access_key}
        await connection.send(json.dumps(hello))
        reply = json.loads(await connection.recv())
        if reply["type"] != "welcome":
            await connection.close()
            raise ConnectionError(f"The signaling server refused {email}: {reply}")
        self._connection = connection
        self._reader = asyncio.create_task(self._read_messages(connection))
        return reply["ice_servers"]

    async def watch(self, email: str, peers: list[str]) -> None:
        await self._send({"type": "watch", "peers": peers})

    async def offer(self, sender: str, receiver: str, sdp: str) -> str:
        """Send the offer, and wait for the answer of ``receiver``.

        Raises:
            ConnectionError: ``receiver`` is offline, or the connection to the
                server closes before the answer comes.
        """
        request_id = uuid.uuid4().hex
        answer = asyncio.get_running_loop().create_future()
        self._pending_answers[request_id] = answer
        try:
            await self._send(
                {"type": "offer", "to": receiver, "id": request_id, "sdp": sdp}
            )
            return await answer
        finally:
            self._pending_answers.pop(request_id, None)

    async def close(self, email: str) -> None:
        for task in self._answer_tasks:
            task.cancel()
        await self._connection.close()
        await self._reader

    async def _read_messages(self, connection: ClientConnection) -> None:
        try:
            async for raw in connection:
                self._handle_message(json.loads(raw))
        except ConnectionClosedError as err:
            logger.error(f"The signaling server closed the connection: {err}")
        finally:
            self._fail_pending_answers()

    def _handle_message(self, message: dict) -> None:
        kind = message["type"]
        if kind == "offer":
            # Each offer gets its own task, so a slow answer does not delay the
            # other messages.
            self._start_answer_task(self._answer_offer(message))
        elif kind == "answer":
            self._resolve_answer(message["id"], message["sdp"])
        elif kind == "error":
            self._handle_error(message)
        elif kind == "online":
            self._on_online(message["email"])
        elif kind != "offline":
            raise ValueError(f"Unexpected message from the signaling server: {kind!r}")

    async def _answer_offer(self, message: dict) -> None:
        sdp = await self._on_offer(message["from"], message["sdp"])
        await self._send(
            {"type": "answer", "to": message["from"], "id": message["id"], "sdp": sdp}
        )

    def _resolve_answer(self, request_id: str, sdp: str) -> None:
        # An offer removes its entry when it no longer waits. Then a late
        # answer finds no entry, and the client ignores it.
        if request_id in self._pending_answers:
            self._pending_answers.pop(request_id).set_result(sdp)

    def _handle_error(self, message: dict) -> None:
        if "id" not in message:
            # An error with no id is a protocol error. The server closes the
            # connection after it.
            logger.error(f"The signaling server reported: {message['reason']}")
        # The client ignores a late error, as it ignores a late answer.
        elif message["id"] in self._pending_answers:
            error = ConnectionError(message["reason"])
            self._pending_answers.pop(message["id"]).set_exception(error)

    def _fail_pending_answers(self) -> None:
        for answer in self._pending_answers.values():
            if not answer.done():
                answer.set_exception(
                    ConnectionError("The connection to the signaling server closed")
                )

    def _start_answer_task(self, coroutine: Coroutine[Any, Any, None]) -> None:
        task = asyncio.create_task(coroutine)
        self._answer_tasks.add(task)
        task.add_done_callback(self._answer_tasks.discard)

    async def _send(self, message: dict) -> None:
        await self._connection.send(json.dumps(message))
