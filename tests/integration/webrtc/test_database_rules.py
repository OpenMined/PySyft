"""firebase/database.rules.json on the Realtime Database emulator: each
signed-in user reads and writes only what the rules allow."""

from collections.abc import Callable
from http import HTTPStatus

import pytest

from syft.sync.connections.webrtc.firebase.database import email_key
from syft.sync.connections.webrtc.firebase.database_peering import (
    acl_path,
    key_bundle_path,
    peer_info_path,
    requests_path,
)
from syft.sync.connections.webrtc.firebase.database_signaling import (
    HELLO,
    HELLO_SLOT,
    OFFER,
    REASON_FIELD,
    REQUEST_FIELD,
    SDP_FIELD,
    SESSION_FIELD,
    TYPE_FIELD,
    inbox_path,
    signal_path,
)
from syft.sync.connections.webrtc.firebase.rest import FirebaseRequestError
from tests.integration.webrtc.firebase_emulators import database_of

ALICE = "alice@example.com"
BOB = "bob@example.com"
CAROL = "carol@example.com"
HELLO_SIGNAL = {TYPE_FIELD: HELLO, SESSION_FIELD: "s1"}
OFFER_SIGNAL = {TYPE_FIELD: OFFER, REQUEST_FIELD: "r1", SDP_FIELD: "v=0"}

pytestmark = pytest.mark.usefixtures("firebase_emulators")


def assert_refused(call: Callable[[], object]) -> None:
    with pytest.raises(FirebaseRequestError) as refusal:
        call()
    assert refusal.value.status == HTTPStatus.UNAUTHORIZED


def test_key_bundle_is_written_by_owner_and_read_by_all(tmp_path):
    alice = database_of(ALICE, tmp_path / "alice")
    bob = database_of(BOB, tmp_path / "bob")

    alice.put(key_bundle_path(ALICE), "{}")

    assert bob.get(key_bundle_path(ALICE)) == "{}"
    assert_refused(lambda: bob.put(key_bundle_path(ALICE), "{}"))
    assert_refused(lambda: bob.get("keys"))


def test_peer_info_is_read_only_by_owner_and_reader(tmp_path):
    alice = database_of(ALICE, tmp_path / "alice")
    bob = database_of(BOB, tmp_path / "bob")
    carol = database_of(CAROL, tmp_path / "carol")

    alice.put(peer_info_path(ALICE, BOB), "{}")

    assert bob.get(peer_info_path(ALICE, BOB)) == "{}"
    assert_refused(lambda: carol.get(peer_info_path(ALICE, BOB)))
    assert_refused(lambda: bob.put(peer_info_path(ALICE, BOB), "{}"))


def test_peer_request_must_name_its_sender(tmp_path):
    alice = database_of(ALICE, tmp_path / "alice")
    bob = database_of(BOB, tmp_path / "bob")
    carol = database_of(CAROL, tmp_path / "carol")

    alice.put(f"{requests_path(BOB)}/{email_key(ALICE)}", ALICE)

    assert bob.get(requests_path(BOB)) == {email_key(ALICE): ALICE}
    assert_refused(lambda: carol.get(requests_path(BOB)))
    forged = f"{requests_path(BOB)}/{email_key(CAROL)}"
    assert_refused(lambda: carol.put(forged, ALICE))
    as_alice = f"{requests_path(BOB)}/{email_key(ALICE)}"
    assert_refused(lambda: carol.put(as_alice, ALICE))


def test_acl_is_written_only_by_owner(tmp_path):
    alice = database_of(ALICE, tmp_path / "alice")
    bob = database_of(BOB, tmp_path / "bob")
    entry = f"{acl_path(ALICE)}/{email_key(BOB)}"

    alice.put(entry, True)

    assert_refused(lambda: bob.put(entry, True))


def test_signal_needs_acceptance_of_receiver(tmp_path):
    alice = database_of(ALICE, tmp_path / "alice")
    bob = database_of(BOB, tmp_path / "bob")

    hello_slot = f"{signal_path(BOB, ALICE)}/{HELLO_SLOT}"
    assert_refused(lambda: alice.put(hello_slot, HELLO_SIGNAL))

    bob.put(f"{acl_path(BOB)}/{email_key(ALICE)}", True)
    alice.put(hello_slot, HELLO_SIGNAL)


def test_offer_with_reason_is_refused(tmp_path):
    alice = database_of(ALICE, tmp_path / "alice")
    bob = database_of(BOB, tmp_path / "bob")
    bob.put(f"{acl_path(BOB)}/{email_key(ALICE)}", True)
    with_reason = {**OFFER_SIGNAL, REASON_FIELD: "any"}

    assert_refused(lambda: alice.post(signal_path(BOB, ALICE), with_reason))


def test_inbox_is_read_and_cleared_only_by_receiver(tmp_path):
    alice = database_of(ALICE, tmp_path / "alice")
    bob = database_of(BOB, tmp_path / "bob")
    carol = database_of(CAROL, tmp_path / "carol")
    bob.put(f"{acl_path(BOB)}/{email_key(ALICE)}", True)
    signal_id = alice.post(signal_path(BOB, ALICE), OFFER_SIGNAL)

    assert bob.get(inbox_path(BOB)) == {email_key(ALICE): {signal_id: OFFER_SIGNAL}}
    assert_refused(lambda: carol.get(inbox_path(BOB)))
    assert_refused(lambda: alice.delete(f"{signal_path(BOB, ALICE)}/{signal_id}"))
    bob.delete(inbox_path(BOB))
    assert bob.get(inbox_path(BOB)) is None


def test_hello_goes_only_to_hello_slot(tmp_path):
    alice = database_of(ALICE, tmp_path / "alice")
    bob = database_of(BOB, tmp_path / "bob")
    bob.put(f"{acl_path(BOB)}/{email_key(ALICE)}", True)
    hello_slot = f"{signal_path(BOB, ALICE)}/{HELLO_SLOT}"

    alice.put(hello_slot, HELLO_SIGNAL)
    alice.put(hello_slot, {**HELLO_SIGNAL, SESSION_FIELD: "s2"})

    assert bob.get(hello_slot) == {**HELLO_SIGNAL, SESSION_FIELD: "s2"}
    assert_refused(lambda: alice.post(signal_path(BOB, ALICE), HELLO_SIGNAL))
    assert_refused(lambda: alice.put(hello_slot, OFFER_SIGNAL))


def test_signal_under_other_sender_is_refused(tmp_path):
    alice = database_of(ALICE, tmp_path / "alice")
    bob = database_of(BOB, tmp_path / "bob")
    bob.put(f"{acl_path(BOB)}/{email_key(CAROL)}", True)

    assert_refused(lambda: alice.post(signal_path(BOB, CAROL), OFFER_SIGNAL))


def test_signal_of_unknown_type_is_refused(tmp_path):
    alice = database_of(ALICE, tmp_path / "alice")
    bob = database_of(BOB, tmp_path / "bob")
    bob.put(f"{acl_path(BOB)}/{email_key(ALICE)}", True)
    goodbye = {TYPE_FIELD: "goodbye", REQUEST_FIELD: "r1", SDP_FIELD: "v=0"}

    assert_refused(lambda: alice.post(signal_path(BOB, ALICE), goodbye))
