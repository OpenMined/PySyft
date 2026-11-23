"""firebase/firestore.rules on the Firestore emulator: each signed-in user reads
and writes only what the rules allow."""

from collections.abc import Callable
from http import HTTPStatus

import pytest

from syft.sync.connections.webrtc.firebase.firestore import SetFields
from syft.sync.connections.webrtc.firebase.firestore_peering import (
    REQUEST_EMAIL_FIELD,
    peer_requests_path,
    shared_path,
)
from syft.sync.connections.webrtc.firebase.firestore_signaling import (
    CREATED_AT_FIELD,
    FROM_FIELD,
    LAST_SEEN_FIELD,
    OFFER,
    REASON_FIELD,
    REQUEST_FIELD,
    SDP_FIELD,
    SESSION_FIELD,
    TYPE_FIELD,
    inbox_path,
    presence_path,
)
from syft.sync.connections.webrtc.firebase.rest import (
    FirebaseRequestError,
    post_json,
)
from tests.integration.webrtc.firebase_emulators import (
    EMULATOR_PROJECT,
    firestore_of,
    sign_in,
)

ALICE = "alice@example.com"
BOB = "bob@example.com"
CAROL = "carol@example.com"

pytestmark = pytest.mark.usefixtures("firebase_emulators")


def assert_forbidden(call: Callable[[], object]) -> None:
    with pytest.raises(FirebaseRequestError) as refusal:
        call()
    assert refusal.value.status == HTTPStatus.FORBIDDEN


def presence(email: str, server_time: bool = True) -> SetFields:
    server_time_fields = (LAST_SEEN_FIELD,) if server_time else ()
    return SetFields(presence_path(email), {SESSION_FIELD: "s1"}, server_time_fields)


def signal(
    receiver: str, sender: str, kind: str = OFFER, extra: dict[str, str] | None = None
) -> SetFields:
    fields = {
        TYPE_FIELD: kind,
        FROM_FIELD: sender,
        REQUEST_FIELD: "r1",
        SDP_FIELD: "v=0",
        **(extra or {}),
    }
    return SetFields(f"{inbox_path(receiver)}/s1", fields, (CREATED_AT_FIELD,))


def list_collection(email: str, folder, collection_id: str) -> None:
    """List a top-level collection as ``email``."""
    session = sign_in(email, folder)
    post_json(
        f"{EMULATOR_PROJECT.firestore_url}/{EMULATOR_PROJECT.documents_name}:runQuery",
        {"structuredQuery": {"from": [{"collectionId": collection_id}]}},
        session.id_token(),
    )


def test_presence_is_written_by_its_user_and_read_by_all(tmp_path):
    alice = firestore_of(ALICE, tmp_path / "alice")
    bob = firestore_of(BOB, tmp_path / "bob")

    alice.write_documents([presence(ALICE)])

    assert bob.get_document(presence_path(ALICE)) is not None
    assert_forbidden(lambda: bob.write_documents([presence(ALICE)]))
    assert_forbidden(lambda: bob.write_documents([presence(BOB, server_time=False)]))


def test_presence_cannot_be_listed(tmp_path):
    firestore_of(ALICE, tmp_path / "alice").write_documents([presence(ALICE)])

    assert_forbidden(lambda: list_collection(BOB, tmp_path / "bob", "presence"))


def test_inbox_is_read_only_by_receiver(tmp_path):
    alice = firestore_of(ALICE, tmp_path / "alice")
    bob = firestore_of(BOB, tmp_path / "bob")
    carol = firestore_of(CAROL, tmp_path / "carol")

    alice.write_documents([signal(BOB, ALICE)])

    received = bob.query_collection(inbox_path(BOB)).documents
    assert [received_signal.text("from") for received_signal in received] == [ALICE]
    assert_forbidden(lambda: carol.query_collection(inbox_path(BOB)))


def test_signal_with_other_sender_or_unknown_type_is_refused(tmp_path):
    alice = firestore_of(ALICE, tmp_path / "alice")

    assert_forbidden(lambda: alice.write_documents([signal(BOB, CAROL)]))
    assert_forbidden(lambda: alice.write_documents([signal(BOB, ALICE, "hello")]))


def test_offer_with_reason_is_refused(tmp_path):
    alice = firestore_of(ALICE, tmp_path / "alice")
    offer = signal(BOB, ALICE, extra={REASON_FIELD: "any"})

    assert_forbidden(lambda: alice.write_documents([offer]))


def test_peer_request_is_read_only_by_its_two_users(tmp_path):
    alice = firestore_of(ALICE, tmp_path / "alice")
    bob = firestore_of(BOB, tmp_path / "bob")
    carol = firestore_of(CAROL, tmp_path / "carol")
    request_path = f"{peer_requests_path(BOB)}/{ALICE}"

    alice.write_documents([SetFields(request_path, {REQUEST_EMAIL_FIELD: ALICE})])

    assert len(bob.query_collection(peer_requests_path(BOB)).documents) == 1
    assert_forbidden(lambda: carol.get_document(request_path))
    assert_forbidden(
        lambda: carol.write_documents(
            [SetFields(request_path, {REQUEST_EMAIL_FIELD: ALICE})]
        )
    )


def test_peer_request_must_name_its_sender(tmp_path):
    carol = firestore_of(CAROL, tmp_path / "carol")
    request_path = f"{peer_requests_path(ALICE)}/{CAROL}"
    forged = SetFields(request_path, {REQUEST_EMAIL_FIELD: BOB})

    assert_forbidden(lambda: carol.write_documents([forged]))
    assert_forbidden(lambda: carol.write_documents([SetFields(request_path, {})]))


def test_shared_document_is_read_only_by_owner_and_reader(tmp_path):
    alice = firestore_of(ALICE, tmp_path / "alice")
    bob = firestore_of(BOB, tmp_path / "bob")
    carol = firestore_of(CAROL, tmp_path / "carol")
    path = shared_path(ALICE, BOB)

    alice.write_documents([SetFields(path, {"key_bundle": "{}"})])

    assert bob.get_document(path).text("key_bundle") == "{}"
    assert_forbidden(lambda: carol.get_document(path))
    assert_forbidden(
        lambda: bob.write_documents([SetFields(path, {"key_bundle": "x"})])
    )
