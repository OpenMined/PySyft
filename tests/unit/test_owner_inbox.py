"""The DO reads every proposal waiting from a peer in one sync. A proposal that
cannot be read or applied is archived with a warning, so it never holds back
the proposals after it."""

import time
import uuid

import pytest

from syft.sync.syftbox_manager import SyftboxManager
from syft.sync.sync.datasite_owner_syncer import DatasiteOwnerSyncer
from syft.sync.transport import MessageKind
from tests.unit.test_sender_binding import send_spoofed_message, sync_do, written_paths


def _pair_with_write_access() -> tuple[SyftboxManager, SyftboxManager]:
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection()
    perms = do_manager.datasite_owner_syncer.perm_context.open(".")
    perms.grant_write_access(ds_manager.email)
    return ds_manager, do_manager


def _send_proposal(ds_manager: SyftboxManager, do_email: str, path: str) -> None:
    send_spoofed_message(ds_manager, do_email, ds_manager.email, path)


def _inbox(do_manager: SyftboxManager, ds_email: str) -> list:
    p2p = do_manager.datasite_owner_syncer.connection_router.transport.p2p
    return p2p.receive(ds_email, MessageKind.PROPOSED_CHANGES)


def test_proposal_that_fails_to_apply_is_archived_and_next_is_applied(monkeypatch):
    ds_manager, do_manager = _pair_with_write_access()
    _send_proposal(ds_manager, do_manager.email, "first.txt")
    _send_proposal(ds_manager, do_manager.email, "second.txt")
    original_handle = DatasiteOwnerSyncer.handle_proposed_filechange_events_message

    def fail_on_first(self, sender_email, message):
        if str(message.proposed_file_changes[0].path_in_datasite) == "first.txt":
            raise RuntimeError("cannot apply")
        return original_handle(self, sender_email, message)

    monkeypatch.setattr(
        DatasiteOwnerSyncer, "handle_proposed_filechange_events_message", fail_on_first
    )
    with pytest.warns(UserWarning, match="applying it failed"):
        sync_do(do_manager)

    assert "second.txt" in written_paths(do_manager)
    assert "first.txt" not in written_paths(do_manager)
    assert _inbox(do_manager, ds_manager.email) == []


def test_unreadable_proposal_is_archived_and_next_is_applied():
    ds_manager, do_manager = _pair_with_write_access()
    older = f"msgv2_{time.time() - 60}_{uuid.uuid4().hex}.tar.gz"
    ds_p2p = ds_manager.datasite_watcher_syncer.connection_router.transport.p2p
    ds_p2p.send(do_manager.email, older, b"garbage", MessageKind.PROPOSED_CHANGES)
    _send_proposal(ds_manager, do_manager.email, "after_garbage.txt")

    with pytest.warns(UserWarning, match=f"Archived message {older} unread"):
        sync_do(do_manager)

    assert "after_garbage.txt" in written_paths(do_manager)
    assert _inbox(do_manager, ds_manager.email) == []
