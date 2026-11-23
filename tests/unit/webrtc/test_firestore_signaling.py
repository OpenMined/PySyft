"""FirestoreSignaling against an in-memory Firestore: the age of offers, the
freshness of presence, and answers that fail."""

import asyncio
from datetime import timedelta

from syft.sync.connections.webrtc.firebase.firestore import SetFields
from syft.sync.connections.webrtc.firebase.firestore_signaling import (
    CREATED_AT_FIELD,
    FROM_FIELD,
    LAST_SEEN_FIELD,
    OFFER,
    OFFER_MAX_AGE,
    PRESENCE_MAX_AGE,
    REQUEST_FIELD,
    SDP_FIELD,
    SESSION_FIELD,
    TYPE_FIELD,
    FirestoreSignaling,
    inbox_path,
    presence_path,
)
from syft.sync.connections.webrtc.firebase.turn import no_ice_servers
from syft.sync.connections.webrtc.real.aiortc_peer_connections import (
    CONNECT_TIMEOUT_SECONDS,
)
from tests.unit.webrtc.fake_firestore import FakeFirestore

DO = "do@example.com"
DS = "ds@example.com"
# Long enough for the first poll of a client that signed in.
FIRST_POLL_SECONDS = 0.2
ONE_SECOND = timedelta(seconds=1)


def offer_from(sender: str, receiver: str) -> SetFields:
    fields = {
        TYPE_FIELD: OFFER,
        FROM_FIELD: sender,
        REQUEST_FIELD: "offer-1",
        SDP_FIELD: "v=0",
    }
    return SetFields(f"{inbox_path(receiver)}/s1", fields, (CREATED_AT_FIELD,))


def presence_of(email: str, session: str) -> SetFields:
    return SetFields(presence_path(email), {SESSION_FIELD: session}, (LAST_SEEN_FIELD,))


def ignore(*args: object) -> None:
    pass


async def answer_offer(sender: str, sdp: str) -> str:
    return "v=0"


def test_offer_max_age_is_below_offerer_timeout():
    # The offerer stops waiting after its connect timeout, so an older offer
    # gets an answer that no one reads.
    assert OFFER_MAX_AGE < timedelta(seconds=CONNECT_TIMEOUT_SECONDS)


def test_offer_older_than_max_age_gets_no_answer():
    firestore = FakeFirestore()
    firestore.write_documents([offer_from(DS, DO)])
    firestore.now += OFFER_MAX_AGE + ONE_SECOND
    offers: list[str] = []

    async def record_offer(sender: str, sdp: str) -> str:
        offers.append(sender)
        return "v=0"

    async def scenario() -> None:
        signaling = FirestoreSignaling(firestore, no_ice_servers)
        await signaling.listen(DO, record_offer, ignore, ignore, ignore)
        await asyncio.sleep(FIRST_POLL_SECONDS)
        await signaling.close(DO)

    asyncio.run(scenario())

    assert offers == []
    assert firestore.query_collection(inbox_path(DO)).documents == []


def test_stale_presence_is_not_reported_online():
    firestore = FakeFirestore()
    firestore.write_documents([presence_of(DO, "s1")])
    firestore.now += PRESENCE_MAX_AGE
    online: list[str] = []

    async def scenario() -> None:
        signaling = FirestoreSignaling(firestore, no_ice_servers)
        await signaling.listen(DS, answer_offer, online.append, ignore, ignore)
        await signaling.watch(DS, [DO])
        await signaling.close(DS)

    asyncio.run(scenario())

    assert online == []


def test_new_presence_session_is_reported_again():
    firestore = FakeFirestore()
    firestore.write_documents([presence_of(DO, "s1")])
    online: list[str] = []

    async def scenario() -> None:
        signaling = FirestoreSignaling(firestore, no_ice_servers)
        await signaling.listen(DS, answer_offer, online.append, ignore, ignore)
        await signaling.watch(DS, [DO])
        await signaling._maybe_poll_presence()
        firestore.write_documents([presence_of(DO, "s2")])
        await signaling._maybe_poll_presence()
        await signaling.close(DS)

    asyncio.run(scenario())

    assert online == [DO, DO]


def test_failed_answer_is_reported():
    firestore = FakeFirestore()
    firestore.write_documents([offer_from(DS, DO)])
    failures: list[ConnectionError] = []

    async def fail_to_answer(sender: str, sdp: str) -> str:
        raise ConnectionError("The TURN function is down")

    async def scenario() -> None:
        signaling = FirestoreSignaling(firestore, no_ice_servers)
        await signaling.listen(DO, fail_to_answer, ignore, ignore, failures.append)
        await asyncio.sleep(FIRST_POLL_SECONDS)
        await signaling.close(DO)

    asyncio.run(scenario())

    [failure] = failures
    assert DS in str(failure) and "TURN function is down" in str(failure)
