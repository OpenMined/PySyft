"""Signaling through the Realtime Database.

A sender writes each offer, answer and refusal at
signal/{receiver}/{sender}/{id}, and its hello at
signal/{receiver}/{sender}/hello, where the next hello replaces it. The rules in firebase/database.rules.json let
it write only if the receiver accepted it. Each client reads its own node,
signal/{me}, as one stream.
"""

import asyncio
import uuid
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any, TypeVar

from syft.sync.connections.exceptions import ConnectionUnavailableError
from syft.sync.connections.webrtc.firebase.database import (
    PATCH,
    DatabaseStream,
    RealtimeDatabase,
    StreamEvent,
    email_key,
    email_of_key,
)
from syft.sync.connections.webrtc.signaling.signaling import (
    IceServerSet,
    IceServerSource,
    OnAnswerFailed,
    OnClosed,
    OnHello,
    OnOffer,
    Signaling,
)

# The fields of a signal, and its types.
TYPE_FIELD = "type"
# The session of the sender of a hello: a new id at each sign-in.
SESSION_FIELD = "session"
# The id of the offer that an offer, an answer or a refusal belongs to.
REQUEST_FIELD = "request"
SDP_FIELD = "sdp"
REASON_FIELD = "reason"
HELLO = "hello"
OFFER = "offer"
ANSWER = "answer"
REFUSE = "refuse"
# The one child of signal/{receiver}/{sender} that holds the sender's hello.
HELLO_SLOT = "hello"

T = TypeVar("T")


def inbox_path(email: str) -> str:
    """Return the node of the signals to ``email``."""
    return f"signal/{email_key(email)}"


def signal_path(receiver: str, sender: str) -> str:
    """Return the node of the signals from ``sender`` to ``receiver``."""
    return f"{inbox_path(receiver)}/{email_key(sender)}"


@dataclass(frozen=True)
class Signal:
    sender_key: str
    signal_id: str
    fields: dict[str, str]

    @property
    def sender(self) -> str:
        return email_of_key(self.sender_key)


def signals_in_event(event: StreamEvent) -> list[Signal]:
    """Return the signals that ``event`` adds to an inbox; a deletion adds
    none.

    Raises:
        ValueError: The event is not at a level of an inbox.
    """
    if event.kind == PATCH:
        changes = [
            (f"{event.path.rstrip('/')}/{key}", value)
            for key, value in event.data.items()
        ]
    else:
        changes = [(event.path, event.data)]
    return [signal for path, value in changes for signal in signals_at(path, value)]


def signals_at(path: str, value: Any) -> list[Signal]:
    """Return the signals in ``value`` at ``path`` of an inbox."""
    parts = [part for part in path.split("/") if part]
    if value is None:
        return []
    if len(parts) == 2:
        return [Signal(parts[0], parts[1], value)]
    if len(parts) == 1:
        return [
            Signal(parts[0], signal_id, fields)
            for signal_id, fields in value.items()
            if fields is not None
        ]
    if not parts:
        return [
            signal
            for sender_key, signals in value.items()
            for signal in signals_at(sender_key, signals)
        ]
    raise ValueError(f"Expected a path of at most two keys in an inbox, got {path!r}")


class DatabaseSignaling(Signaling):
    """``Signaling`` for one email, through the Realtime Database.

    ``listen`` opens the stream of the inbox, and deletes the signals that came
    before it: no sender waits for them now. Each later signal is handled, then
    deleted; each offer, answer and refusal only once. If the stream fails before ``close``, the session ends, and
    ``on_closed`` gets the reason.
    """

    # listen() sets these.
    _email: str
    # A new id at each listen. A peer that sees a new id knows that this
    # client signed in again.
    _session: str
    _loop: asyncio.AbstractEventLoop
    _stream: DatabaseStream
    # Done when the first event of the stream, what came before, is handled.
    _started: asyncio.Future[None]
    _on_offer: OnOffer
    _on_hello: OnHello
    _on_closed: OnClosed
    _on_answer_failed: OnAnswerFailed

    def __init__(self, database: RealtimeDatabase, ice_server_source: IceServerSource):
        self._database = database
        self._ice_server_source = ice_server_source
        # True from the start of close(). The end of the session is then no
        # error.
        self._closing = False
        # The ids of the signals that this session handled.
        self._handled_ids: set[str] = set()
        # offer id -> the reply of the peer to the offer
        self._pending_replies: dict[str, asyncio.Future[str]] = {}
        # The tasks that handle signals and answer offers, until they end. This
        # set keeps a strong reference to each task, because asyncio keeps only
        # a weak one.
        self._tasks: set[asyncio.Task] = set()
        # signal type -> the method that handles it
        self._signal_handlers: dict[str, Callable[[Signal], None]] = {
            HELLO: self._handle_hello,
            OFFER: self._handle_offer,
            ANSWER: self._handle_answer,
            REFUSE: self._handle_refuse,
        }

    async def listen(
        self,
        email: str,
        on_offer: OnOffer,
        on_hello: OnHello,
        on_closed: OnClosed,
        on_answer_failed: OnAnswerFailed,
    ) -> None:
        """Open the stream of the inbox of ``email``.

        Raises:
            ConnectionError: The database refuses the stream, or cannot be
                reached.
        """
        self._email = email
        self._session = uuid.uuid4().hex
        self._on_offer = on_offer
        self._on_hello = on_hello
        self._on_closed = on_closed
        self._on_answer_failed = on_answer_failed
        self._closing = False
        self._handled_ids = set()
        self._loop = asyncio.get_running_loop()
        self._started = self._loop.create_future()
        self._stream = await self._call_database(
            self._database.stream, inbox_path(email), self._on_event, self._on_end
        )
        await self._started

    async def ice_servers(self, email: str) -> IceServerSet:
        """Raises:
        ConnectionError: The ICE server source refuses the user, or cannot be
            reached.
        """
        try:
            return await self._ice_server_source()
        except ConnectionUnavailableError as err:
            raise ConnectionError(str(err)) from err

    async def hello(self, email: str, peers: list[str]) -> None:
        """Raises:
        ConnectionError: The database refuses a hello, as it does to a peer
            that did not accept ``email``, or cannot be reached.
        """
        hello = {TYPE_FIELD: HELLO, SESSION_FIELD: self._session}
        for peer in peers:
            hello_path = f"{signal_path(peer, self._email)}/{HELLO_SLOT}"
            await self._call_database(self._database.put, hello_path, hello)

    async def offer(self, sender: str, receiver: str, sdp: str) -> str:
        """Send the offer, and wait for the reply of ``receiver``.

        Raises:
            ConnectionRefusedError: ``receiver`` refuses the offer.
            ConnectionError: The session ends before the reply comes.
        """
        offer_id = uuid.uuid4().hex
        reply = self._loop.create_future()
        self._pending_replies[offer_id] = reply
        try:
            offer = {TYPE_FIELD: OFFER, REQUEST_FIELD: offer_id, SDP_FIELD: sdp}
            await self._send_signal(receiver, offer)
            return await reply
        finally:
            self._pending_replies.pop(offer_id, None)

    async def close(self, email: str) -> None:
        self._closing = True
        for task in self._tasks:
            task.cancel()
        await asyncio.to_thread(self._stream.close)

    def _on_event(self, event: StreamEvent) -> None:
        """Runs on the thread of the stream."""
        self._loop.call_soon_threadsafe(self._handle_event, event)

    def _on_end(self, error: OSError) -> None:
        """Runs on the thread of the stream."""
        self._loop.call_soon_threadsafe(self._maybe_end_session, error)

    def _handle_event(self, event: StreamEvent) -> None:
        # Tasks start in the order of the events.
        self._start_task(self._process_event(event))

    async def _process_event(self, event: StreamEvent) -> None:
        signals = signals_in_event(event)
        if self._started.done():
            await self._handle_signals(signals)
        else:
            self._started.set_result(None)
            await self._maybe_delete_signals(signals)

    async def _handle_signals(self, signals: list[Signal]) -> None:
        # A stream opened again after its ID token expired starts with what is
        # in the inbox, so a signal can come twice. A hello can come again in
        # its slot, and handling it twice does no harm.
        new_signals = [
            s
            for s in signals
            if s.signal_id == HELLO_SLOT or s.signal_id not in self._handled_ids
        ]
        self._handled_ids.update(
            s.signal_id for s in new_signals if s.signal_id != HELLO_SLOT
        )
        await self._maybe_delete_signals(new_signals)
        for signal in new_signals:
            self._signal_handlers[signal.fields[TYPE_FIELD]](signal)

    async def _maybe_delete_signals(self, signals: list[Signal]) -> None:
        if signals:
            deletes = {
                f"{inbox_path(self._email)}/{s.sender_key}/{s.signal_id}": None
                for s in signals
            }
            await self._call_database(self._database.update, deletes)

    def _handle_hello(self, signal: Signal) -> None:
        self._on_hello(signal.sender, signal.fields[SESSION_FIELD])

    def _handle_offer(self, signal: Signal) -> None:
        task = self._start_task(self._answer_offer(signal), ends_session=False)
        task.add_done_callback(
            lambda done: self._maybe_report_failed_answer(done, signal.sender)
        )

    async def _answer_offer(self, signal: Signal) -> None:
        reply = {REQUEST_FIELD: signal.fields[REQUEST_FIELD]}
        try:
            sdp = await self._on_offer(signal.sender, signal.fields[SDP_FIELD])
        except ConnectionRefusedError as err:
            refusal = {**reply, TYPE_FIELD: REFUSE, REASON_FIELD: str(err)}
            await self._send_signal(signal.sender, refusal)
        else:
            answer = {**reply, TYPE_FIELD: ANSWER, SDP_FIELD: sdp}
            await self._send_signal(signal.sender, answer)

    def _maybe_report_failed_answer(
        self, answer_task: asyncio.Task, sender: str
    ) -> None:
        """Give the reason to ``on_answer_failed`` if ``answer_task`` failed."""
        if answer_task.cancelled() or answer_task.exception() is None:
            return
        self._on_answer_failed(
            ConnectionError(
                f"Could not answer the offer of {sender}: {answer_task.exception()!r}"
            )
        )

    def _handle_answer(self, signal: Signal) -> None:
        # An offer removes its entry when it no longer waits. Then a late reply
        # finds no entry, and the client ignores it.
        offer_id = signal.fields[REQUEST_FIELD]
        if offer_id in self._pending_replies:
            self._pending_replies.pop(offer_id).set_result(signal.fields[SDP_FIELD])

    def _handle_refuse(self, signal: Signal) -> None:
        offer_id = signal.fields[REQUEST_FIELD]
        if offer_id in self._pending_replies:
            refusal = ConnectionRefusedError(signal.fields[REASON_FIELD])
            self._pending_replies.pop(offer_id).set_exception(refusal)

    def _maybe_end_session(self, error: BaseException) -> None:
        """Fail what waits on the session, stop the stream, and give the reason
        to ``on_closed``, unless the session is closing.

        Before the first event of the stream, only ``listen`` raises: no session
        has started, and the stream thread has ended.
        """
        if self._closing:
            return
        self._closing = True
        ended = ConnectionError(f"The signaling session ended: {error!r}")
        if not self._started.done():
            self._started.set_exception(ended)
            return
        for reply in self._pending_replies.values():
            if not reply.done():
                reply.set_exception(ended)
        self._loop.run_in_executor(None, self._stream.close)
        self._on_closed(ended)

    def _start_task(
        self, coroutine: Coroutine[Any, Any, None], ends_session: bool = True
    ) -> asyncio.Task:
        """Run ``coroutine`` as a task. If ``ends_session``, its failure ends the
        session."""
        task = asyncio.create_task(coroutine)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        if ends_session:
            task.add_done_callback(self._maybe_end_session_on_failure)
        return task

    def _maybe_end_session_on_failure(self, task: asyncio.Task) -> None:
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            self._maybe_end_session(error)

    async def _send_signal(self, receiver: str, fields: dict[str, str]) -> None:
        await self._call_database(
            self._database.post, signal_path(receiver, self._email), fields
        )

    async def _call_database(self, call: Callable[..., T], *args: Any) -> T:
        """Run a blocking database call on a worker thread.

        Raises:
            ConnectionError: The database refuses the call, or cannot be
                reached.
        """
        try:
            return await asyncio.to_thread(call, *args)
        except ConnectionUnavailableError as err:
            raise ConnectionError(str(err)) from err
