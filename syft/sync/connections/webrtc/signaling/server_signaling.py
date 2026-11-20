"""Signaling through the syft signaling server in ``server.py``, over one
WebSocket."""

import asyncio
import json
import uuid
from collections.abc import Callable, Coroutine
from typing import Any

from websockets.asyncio.client import ClientConnection, connect

from syft.sync.connections.webrtc.signaling.signaling import (
    IceServers,
    OnClosed,
    OnOffer,
    OnOnline,
    Signaling,
)
from syft.sync.connections.webrtc.webrtc_config import WebRTCConfig


class ServerSignaling(Signaling):
    """``Signaling`` for one email, through the signaling server.

    ``listen`` signs in to the server and keeps the WebSocket open until
    ``close``. If the session ends before ``close``, ``on_closed`` gets the
    reason.
    """

    # listen() sets these.
    _connection: ClientConnection
    _reader: asyncio.Task
    _on_offer: OnOffer
    _on_online: OnOnline
    _on_closed: OnClosed

    def __init__(self, config: WebRTCConfig):
        self._config = config
        # True from the start of close(). The end of the session is then no
        # error.
        self._closing = False
        # The reason in the last error with no request id, before the server
        # closes the connection.
        self._server_error: str | None = None
        # request id -> the reply to a request of this client
        self._pending_replies: dict[str, asyncio.Future[Any]] = {}
        # The tasks that answer offers, until they end. This set keeps a strong
        # reference to each task, because asyncio keeps only a weak one.
        self._answer_tasks: set[asyncio.Task] = set()
        # message type -> the method that handles it
        self._message_handlers: dict[str, Callable[[dict], None]] = {
            "offer": self._handle_offer,
            "answer": self._handle_answer,
            "refuse": self._handle_refuse,
            "ice_servers": self._handle_ice_servers,
            "error": self._handle_error,
            "online": self._handle_online,
            "offline": self._handle_offline,
        }

    async def listen(
        self, email: str, on_offer: OnOffer, on_online: OnOnline, on_closed: OnClosed
    ) -> None:
        """Sign in to the server as ``email``.

        Raises:
            ConnectionError: The server refuses the sign-in.
        """
        self._on_offer = on_offer
        self._on_online = on_online
        self._on_closed = on_closed
        self._closing = False
        self._server_error = None
        connection = await connect(self._config.signaling_url)
        hello = {"type": "hello", "email": email, "access_key": self._config.access_key}
        await connection.send(json.dumps(hello))
        reply = json.loads(await connection.recv())
        if reply["type"] != "welcome":
            await connection.close()
            raise ConnectionError(f"The signaling server refused {email}: {reply}")
        self._connection = connection
        self._reader = asyncio.create_task(self._read_messages(connection))
        self._reader.add_done_callback(self._report_end_of_session)

    async def ice_servers(self, email: str) -> IceServers:
        """Ask the server for ICE servers with new short-lived TURN credentials.

        Raises:
            ConnectionError: The session ends before the reply comes.
        """
        return await self._request({"type": "get_ice_servers"})

    async def watch(self, email: str, peers: list[str]) -> None:
        await self._send({"type": "watch", "peers": peers})

    async def offer(self, sender: str, receiver: str, sdp: str) -> str:
        """Send the offer, and wait for the answer of ``receiver``.

        Raises:
            ConnectionRefusedError: ``receiver`` refuses the offer.
            ConnectionError: ``receiver`` is offline, or the session ends
                before the answer comes.
        """
        return await self._request({"type": "offer", "to": receiver, "sdp": sdp})

    async def close(self, email: str) -> None:
        self._closing = True
        for task in self._answer_tasks:
            task.cancel()
        await self._connection.close()
        await self._reader

    async def _request(self, message: dict) -> Any:
        """Send ``message`` with a new request id, and return the value of the
        reply."""
        request_id = uuid.uuid4().hex
        reply = asyncio.get_running_loop().create_future()
        self._pending_replies[request_id] = reply
        try:
            await self._send({**message, "id": request_id})
            return await reply
        finally:
            self._pending_replies.pop(request_id, None)

    async def _read_messages(self, connection: ClientConnection) -> None:
        async for raw in connection:
            self._handle_message(json.loads(raw))

    def _report_end_of_session(self, reader: asyncio.Task) -> None:
        """Fail the requests that wait for a reply. If the session ended
        before ``close``, give the reason to ``on_closed``."""
        error = ConnectionError(
            f"The signaling session ended: {self._end_reason(reader)}"
        )
        for reply in self._pending_replies.values():
            if not reply.done():
                reply.set_exception(error)
        if not self._closing:
            self._on_closed(error)

    def _end_reason(self, reader: asyncio.Task) -> str:
        if reader.cancelled():
            return "the client stopped reading"
        if reader.exception() is not None:
            return str(reader.exception())
        reason = (
            f"the server closed the connection with code "
            f"{self._connection.close_code} ({self._connection.close_reason})"
        )
        if self._server_error is not None:
            reason += f" after the error: {self._server_error}"
        return reason

    def _handle_message(self, message: dict) -> None:
        kind = message["type"]
        if kind not in self._message_handlers:
            expected = ", ".join(self._message_handlers)
            raise ValueError(
                f"Expected a message of type {expected} from the signaling server, "
                f"got {kind!r}"
            )
        self._message_handlers[kind](message)

    def _handle_offer(self, message: dict) -> None:
        # Each offer gets its own task, so a slow answer does not delay the
        # other messages.
        self._start_answer_task(self._answer_offer(message))

    async def _answer_offer(self, message: dict) -> None:
        reply = {"to": message["from"], "id": message["id"]}
        try:
            sdp = await self._on_offer(message["from"], message["sdp"])
        except ConnectionRefusedError as err:
            # The server gives the reason to the offerer.
            await self._send({**reply, "type": "refuse", "reason": str(err)})
        else:
            await self._send({**reply, "type": "answer", "sdp": sdp})

    def _handle_answer(self, message: dict) -> None:
        self._resolve_reply(message["id"], message["sdp"])

    def _handle_ice_servers(self, message: dict) -> None:
        self._resolve_reply(message["id"], message["ice_servers"])

    def _handle_refuse(self, message: dict) -> None:
        self._reject_reply(message["id"], ConnectionRefusedError(message["reason"]))

    def _handle_error(self, message: dict) -> None:
        if "id" not in message:
            # A protocol error. The server closes the connection after it.
            self._server_error = message["reason"]
        else:
            self._reject_reply(message["id"], ConnectionError(message["reason"]))

    def _handle_online(self, message: dict) -> None:
        self._on_online(message["email"])

    def _handle_offline(self, message: dict) -> None:
        """Ignore the message: the signaling has nothing to do for a peer that
        goes offline."""

    def _resolve_reply(self, request_id: str, value: Any) -> None:
        # A request removes its entry when it no longer waits. Then a late reply
        # finds no entry, and the client ignores it.
        if request_id in self._pending_replies:
            self._pending_replies.pop(request_id).set_result(value)

    def _reject_reply(self, request_id: str, error: ConnectionError) -> None:
        # The client ignores a late error, as it ignores a late reply.
        if request_id in self._pending_replies:
            self._pending_replies.pop(request_id).set_exception(error)

    def _start_answer_task(self, coroutine: Coroutine[Any, Any, None]) -> None:
        task = asyncio.create_task(coroutine)
        self._answer_tasks.add(task)
        task.add_done_callback(self._answer_tasks.discard)

    async def _send(self, message: dict) -> None:
        await self._connection.send(json.dumps(message))
