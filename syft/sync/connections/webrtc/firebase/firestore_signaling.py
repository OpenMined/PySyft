"""Signaling through Firestore.

Each client has a presence document and an inbox of signals: offers, answers
and refusals. Each client polls its inbox and the presence of the peers it
watches. firebase/firestore.rules has the paths and who may read and write
them.
"""

import asyncio
import uuid
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any, TypeVar

from syft.sync.connections.exceptions import ConnectionUnavailableError
from syft.sync.connections.webrtc.firebase.firestore import (
    DeleteDocument,
    FirestoreClient,
    FirestoreDocument,
    SetFields,
)
from syft.sync.connections.webrtc.signaling.signaling import (
    IceServers,
    IceServerSource,
    OnAnswerFailed,
    OnClosed,
    OnOffer,
    OnOnline,
    Signaling,
)

INBOX_POLL_SECONDS = 1.0
PRESENCE_POLL_SECONDS = 5.0
HEARTBEAT_SECONDS = 30.0
# A peer that missed a heartbeat is offline.
PRESENCE_MAX_AGE = timedelta(seconds=2 * HEARTBEAT_SECONDS)
# The offerer stops waiting for an answer after its connect timeout of 20 s.
# A client drops an older offer, because no one reads its answer.
OFFER_MAX_AGE = timedelta(seconds=10)

# The fields of a presence document.
SESSION_FIELD = "session"
LAST_SEEN_FIELD = "last_seen"
# The fields of a signal, and its types.
TYPE_FIELD = "type"
FROM_FIELD = "from"
# The id of the offer that the signal belongs to.
REQUEST_FIELD = "request"
SDP_FIELD = "sdp"
REASON_FIELD = "reason"
CREATED_AT_FIELD = "created_at"
OFFER = "offer"
ANSWER = "answer"
REFUSE = "refuse"

T = TypeVar("T")


def presence_path(email: str) -> str:
    return f"presence/{email.lower()}"


def inbox_path(email: str) -> str:
    return f"inbox/{email.lower()}/signals"


class FirestoreSignaling(Signaling):
    """``Signaling`` for one email, through Firestore.

    ``listen`` writes the presence of the email, and starts a task that polls
    the inbox and the presence of the watched peers. The task writes the
    presence again each ``HEARTBEAT_SECONDS``. If a poll fails before
    ``close``, the session ends, and ``on_closed`` gets the reason.
    """

    # listen() sets these.
    _email: str
    # A new id at each listen. A peer that sees a new id knows that this
    # client signed in again.
    _session: str
    _poll_task: asyncio.Task
    _on_offer: OnOffer
    _on_online: OnOnline
    _on_closed: OnClosed
    _on_answer_failed: OnAnswerFailed

    def __init__(self, firestore: FirestoreClient, ice_server_source: IceServerSource):
        self._firestore = firestore
        self._ice_server_source = ice_server_source
        # True from the start of close(). The end of the session is then no
        # error.
        self._closing = False
        self._watched: list[str] = []
        # watched peer in lower case -> the presence session of the peer when
        # on_online last got it
        self._reported_sessions: dict[str, str] = {}
        # offer id -> the reply of the peer to the offer
        self._pending_replies: dict[str, asyncio.Future[str]] = {}
        # The tasks that answer offers, until they end. This set keeps a strong
        # reference to each task, because asyncio keeps only a weak one.
        self._answer_tasks: set[asyncio.Task] = set()
        # signal type -> the method that handles it
        self._signal_handlers: dict[
            str, Callable[[FirestoreDocument, datetime], None]
        ] = {
            OFFER: self._handle_offer,
            ANSWER: self._handle_answer,
            REFUSE: self._handle_refuse,
        }

    async def listen(
        self,
        email: str,
        on_offer: OnOffer,
        on_online: OnOnline,
        on_closed: OnClosed,
        on_answer_failed: OnAnswerFailed,
    ) -> None:
        """Write the presence of ``email``, and start to poll.

        Raises:
            ConnectionError: Firestore refuses the presence, or cannot be
                reached.
        """
        self._email = email
        self._session = uuid.uuid4().hex
        self._on_offer = on_offer
        self._on_online = on_online
        self._on_closed = on_closed
        self._on_answer_failed = on_answer_failed
        self._closing = False
        await self._write_presence()
        self._poll_task = asyncio.create_task(self._poll_until_closed())
        self._poll_task.add_done_callback(self._report_end_of_session)

    async def ice_servers(self, email: str) -> IceServers:
        """Raises:
        ConnectionError: The ICE server source refuses the user, or cannot be
            reached.
        """
        try:
            return await self._ice_server_source()
        except ConnectionUnavailableError as err:
            raise ConnectionError(str(err)) from err

    async def watch(self, email: str, peers: list[str]) -> None:
        self._watched = peers
        # Thus the first poll reports each peer that is online now.
        self._reported_sessions = {}
        await self._maybe_poll_presence()

    async def offer(self, sender: str, receiver: str, sdp: str) -> str:
        """Send the offer, and wait for the reply of ``receiver``.

        Raises:
            ConnectionRefusedError: ``receiver`` refuses the offer.
            ConnectionError: The session ends before the reply comes.
        """
        offer_id = uuid.uuid4().hex
        reply = asyncio.get_running_loop().create_future()
        self._pending_replies[offer_id] = reply
        try:
            offer = {TYPE_FIELD: OFFER, REQUEST_FIELD: offer_id, SDP_FIELD: sdp}
            await self._send_signal(receiver, offer)
            return await reply
        finally:
            self._pending_replies.pop(offer_id, None)

    async def close(self, email: str) -> None:
        self._closing = True
        for task in self._answer_tasks:
            task.cancel()
        self._poll_task.cancel()
        await asyncio.wait([self._poll_task])
        await self._call_firestore(
            self._firestore.write_documents, [DeleteDocument(presence_path(email))]
        )

    async def _poll_until_closed(self) -> None:
        loop = asyncio.get_running_loop()
        next_presence_poll = loop.time()
        next_heartbeat = loop.time() + HEARTBEAT_SECONDS
        while True:
            await self._poll_inbox()
            if loop.time() >= next_presence_poll:
                await self._maybe_poll_presence()
                next_presence_poll = loop.time() + PRESENCE_POLL_SECONDS
            if loop.time() >= next_heartbeat:
                await self._write_presence()
                next_heartbeat = loop.time() + HEARTBEAT_SECONDS
            await asyncio.sleep(INBOX_POLL_SECONDS)

    async def _poll_inbox(self) -> None:
        read = await self._call_firestore(
            self._firestore.query_collection, inbox_path(self._email)
        )
        if read.documents:
            deletes = [DeleteDocument(signal.path) for signal in read.documents]
            await self._call_firestore(self._firestore.write_documents, deletes)
        for signal in read.documents:
            self._signal_handlers[signal.text(TYPE_FIELD)](signal, read.read_time)

    async def _maybe_poll_presence(self) -> None:
        """Call ``on_online`` for each watched peer that came online, or signed
        in again, since the last poll."""
        peers = list(self._watched)
        if not peers:
            return
        read = await self._call_firestore(
            self._firestore.get_documents, [presence_path(peer) for peer in peers]
        )
        presences = {presence.path: presence for presence in read.documents}
        for peer in peers:
            presence = presences.get(presence_path(peer))
            if presence is None or not is_fresh(presence, read.read_time):
                self._reported_sessions.pop(peer.lower(), None)
            elif self._reported_sessions.get(peer.lower()) != presence.text(
                SESSION_FIELD
            ):
                self._reported_sessions[peer.lower()] = presence.text(SESSION_FIELD)
                self._on_online(peer)

    def _handle_offer(self, signal: FirestoreDocument, read_time: datetime) -> None:
        # aioice of the offerer has given up on an older offer.
        if read_time - signal.time(CREATED_AT_FIELD) < OFFER_MAX_AGE:
            self._start_answer_task(signal)

    async def _answer_offer(self, signal: FirestoreDocument) -> None:
        sender = signal.text(FROM_FIELD)
        reply = {REQUEST_FIELD: signal.text(REQUEST_FIELD)}
        try:
            sdp = await self._on_offer(sender, signal.text(SDP_FIELD))
        except ConnectionRefusedError as err:
            refusal = {**reply, TYPE_FIELD: REFUSE, REASON_FIELD: str(err)}
            await self._send_signal(sender, refusal)
        else:
            await self._send_signal(
                sender, {**reply, TYPE_FIELD: ANSWER, SDP_FIELD: sdp}
            )

    def _handle_answer(self, signal: FirestoreDocument, read_time: datetime) -> None:
        # An offer removes its entry when it no longer waits. Then a late reply
        # finds no entry, and the client ignores it.
        offer_id = signal.text(REQUEST_FIELD)
        if offer_id in self._pending_replies:
            self._pending_replies.pop(offer_id).set_result(signal.text(SDP_FIELD))

    def _handle_refuse(self, signal: FirestoreDocument, read_time: datetime) -> None:
        offer_id = signal.text(REQUEST_FIELD)
        if offer_id in self._pending_replies:
            refusal = ConnectionRefusedError(signal.text(REASON_FIELD))
            self._pending_replies.pop(offer_id).set_exception(refusal)

    def _report_end_of_session(self, poll_task: asyncio.Task) -> None:
        """Fail the offers that wait for a reply. If the session ended before
        ``close``, give the reason to ``on_closed``."""
        if poll_task.cancelled():
            reason = "the client stopped polling"
        else:
            reason = repr(poll_task.exception())
        error = ConnectionError(f"The signaling session ended: {reason}")
        for reply in self._pending_replies.values():
            if not reply.done():
                reply.set_exception(error)
        if not self._closing:
            self._on_closed(error)

    async def _write_presence(self) -> None:
        presence = SetFields(
            presence_path(self._email),
            {SESSION_FIELD: self._session},
            server_time_fields=(LAST_SEEN_FIELD,),
        )
        await self._call_firestore(self._firestore.write_documents, [presence])

    async def _send_signal(self, receiver: str, fields: dict[str, str]) -> None:
        signal = SetFields(
            f"{inbox_path(receiver)}/{uuid.uuid4().hex}",
            {**fields, FROM_FIELD: self._email},
            server_time_fields=(CREATED_AT_FIELD,),
        )
        await self._call_firestore(self._firestore.write_documents, [signal])

    def _start_answer_task(self, offer: FirestoreDocument) -> None:
        task = asyncio.create_task(self._answer_offer(offer))
        self._answer_tasks.add(task)
        task.add_done_callback(self._answer_tasks.discard)
        sender = offer.text(FROM_FIELD)
        task.add_done_callback(
            lambda done: self._maybe_report_failed_answer(done, sender)
        )

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

    async def _call_firestore(self, call: Callable[..., T], *args: Any) -> T:
        """Run a blocking Firestore call on a worker thread.

        Raises:
            ConnectionError: Firestore refuses the call, or cannot be reached.
        """
        try:
            return await asyncio.to_thread(call, *args)
        except ConnectionUnavailableError as err:
            raise ConnectionError(str(err)) from err


def is_fresh(presence: FirestoreDocument, read_time: datetime) -> bool:
    """Return whether ``presence`` had a heartbeat recently at ``read_time``."""
    return read_time - presence.time(LAST_SEEN_FIELD) < PRESENCE_MAX_AGE
