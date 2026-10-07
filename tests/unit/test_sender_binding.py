"""The sender of a proposed change is the peer whose inbox held the message.

The message body carries a `sender_email` field that the sender writes. The DO
must never read identity from that field.

The honest path is covered by test_sync_manager.py, which fails in six places
when this binding breaks.
"""

from syft.sync.messages.proposed_filechange import (
    ProposedFileChange,
    ProposedFileChangesMessage,
)
from syft.sync.syftbox_manager import SyftboxManager


def send_spoofed_message(
    ds_manager: SyftboxManager, do_email: str, claimed_email: str, path: str
) -> None:
    """Put a message in the DO's inbox that claims another sender."""
    message = ProposedFileChangesMessage(
        sender_email=claimed_email,
        proposed_file_changes=[
            ProposedFileChange(
                old_hash=None,
                path_in_datasite=path,
                content="written by the DS",
                datasite_email=do_email,
            )
        ],
    )
    router = ds_manager.datasite_watcher_syncer.connection_router
    router.watcher_send_proposed_file_changes_message(do_email, message)


def sync_do(do_manager: SyftboxManager) -> None:
    """Sync the DO. A reply to a spoofed address fails, and that is not the point
    of these tests: it happens after the write, so the assertion still reads what
    landed. A DO that drops the message makes no reply and raises nothing."""
    try:
        do_manager.sync()
    except ValueError as exc:
        if "Outbox folder" not in str(exc):
            raise


def written_paths(do_manager: SyftboxManager) -> list[str]:
    connection = do_manager.datasite_owner_syncer.event_cache.file_connection
    return [str(path) for path, _ in connection.get_items()]


def test_sender_email_in_body_cannot_claim_the_datasite_owner():
    """A DS that claims the DO's email gets no write access.

    `ACLService.can_access` returns True for the datasite owner, so this claim
    would otherwise bypass every rule in the datasite.
    """
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection()
    do_email = do_manager.email
    path = "owner_only.txt"

    # The DS holds no write grant on this path.
    send_spoofed_message(ds_manager, do_email, do_email, path)
    sync_do(do_manager)

    assert path not in written_paths(do_manager)


def test_spoofed_sender_does_not_name_an_archive_folder():
    """The DO archives a handled message under the peer that sent it.

    `owner_remove_proposed_filechange_message_from_inbox` builds the archive
    folder name from an email, and creates the folder. Read that email from the
    transport, so a claim in the body cannot name a folder in the DO's Drive.
    """
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection()
    do_email = do_manager.email
    claimed_email = "stranger@openmined.org"

    send_spoofed_message(ds_manager, do_email, claimed_email, "any.txt")
    sync_do(do_manager)

    connection = do_manager.datasite_owner_syncer.connection_router.connections[0]
    folder_names = [
        f.name for f in connection.drive_service._backing_store.files.values()
    ]
    assert not [name for name in folder_names if claimed_email in name]


def test_sender_email_case_does_not_refuse_an_honest_message():
    """The two emails are compared without case, as email addresses are.

    An exact compare refuses the message, and the DS loses the write it has a
    grant for.
    """
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection()
    ds_email = ds_manager.email
    do_email = do_manager.email
    path = "mixed_case.txt"

    do_manager.datasite_owner_syncer.perm_context.open(".").grant_write_access(ds_email)

    send_spoofed_message(ds_manager, do_email, ds_email.upper(), path)
    sync_do(do_manager)

    assert path in written_paths(do_manager)
