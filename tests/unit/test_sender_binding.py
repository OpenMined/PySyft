"""The sender of a proposed change is the peer whose inbox held the message.

The message body carries a `sender_email` field that the sender writes. The DO
must never use that field to decide write access.
"""

from syft.sync.messages.proposed_filechange import (
    ProposedFileChange,
    ProposedFileChangesMessage,
)
from syft.sync.syftbox_manager import SyftboxManager


def proposed_message(
    sender_email: str, do_email: str, path: str, content: str
) -> ProposedFileChangesMessage:
    return ProposedFileChangesMessage(
        sender_email=sender_email,
        proposed_file_changes=[
            ProposedFileChange(
                old_hash=None,
                path_in_datasite=path,
                content=content,
                datasite_email=do_email,
            )
        ],
    )


def send_to_inbox(
    ds_manager: SyftboxManager, do_email: str, message: ProposedFileChangesMessage
) -> None:
    """Put `message` in the DO's inbox for this DS, over the real transport."""
    router = ds_manager.datasite_watcher_syncer.connection_router
    router.watcher_send_proposed_file_changes_message(do_email, message)


def written_paths(do_manager: SyftboxManager) -> list[str]:
    connection = do_manager.datasite_owner_syncer.event_cache.file_connection
    return [str(path) for path, _ in connection.get_items()]


def sync_do(do_manager: SyftboxManager) -> None:
    """Sync the DO, and tolerate a failed reply to a spoofed address.

    A spoofed message names a sender with no outbox, so the acknowledgement
    raises. That happens after the write, so the test still reads what landed.
    Once the DO drops such a message there is no reply and no exception.
    """
    try:
        do_manager.sync()
    except ValueError as exc:
        if "Outbox folder" not in str(exc):
            raise


def test_sender_email_in_body_cannot_claim_the_datasite_owner():
    """A DS that claims the DO's email gets no write access.

    `ACLService.can_access` returns True for the datasite owner, so this claim
    would otherwise bypass every rule in the datasite.
    """
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection()
    do_email = do_manager.email
    path = "owner_only.txt"

    # The DS holds no write grant on this path.
    send_to_inbox(
        ds_manager,
        do_email,
        proposed_message(do_email, do_email, path, "written by the DS"),
    )
    sync_do(do_manager)

    assert path not in written_paths(do_manager)


def test_sender_email_in_body_cannot_claim_another_peer():
    """A DS that claims a second DS's email does not get that DS's grants."""
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection()
    do_email = do_manager.email
    other_ds_email = "other@openmined.org"
    path = "for_other_ds.txt"

    do_manager.datasite_owner_syncer.perm_context.open(".").grant_write_access(
        other_ds_email
    )

    send_to_inbox(
        ds_manager,
        do_email,
        proposed_message(other_ds_email, do_email, path, "written by the DS"),
    )
    sync_do(do_manager)

    assert path not in written_paths(do_manager)


def test_honest_sender_still_writes():
    """A message whose body matches the transport peer is applied as before."""
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection()
    ds_email = ds_manager.email
    do_email = do_manager.email
    path = "granted.txt"

    do_manager.datasite_owner_syncer.perm_context.open(".").grant_write_access(ds_email)

    send_to_inbox(
        ds_manager, do_email, proposed_message(ds_email, do_email, path, "hello")
    )
    sync_do(do_manager)

    assert path in written_paths(do_manager)


def test_sender_email_case_does_not_refuse_an_honest_message():
    """The two emails are compared without case, as email addresses are."""
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection()
    ds_email = ds_manager.email
    do_email = do_manager.email
    path = "mixed_case.txt"

    do_manager.datasite_owner_syncer.perm_context.open(".").grant_write_access(ds_email)

    send_to_inbox(
        ds_manager,
        do_email,
        proposed_message(ds_email.upper(), do_email, path, "hello"),
    )
    sync_do(do_manager)

    assert path in written_paths(do_manager)


def drive_folder_names(manager: SyftboxManager) -> list[str]:
    connection = manager.datasite_owner_syncer.connection_router.connections[0]
    store = connection.drive_service._backing_store
    return [f.name for f in store.files.values()]


def test_spoofed_sender_does_not_create_an_archive_folder():
    """The DO archives a handled message under the peer that sent it.

    `owner_remove_proposed_filechange_message_from_inbox` builds the archive
    folder name from an email. Read that email from the transport, so a claim in
    the body cannot name the folder or create one.
    """
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection()
    do_email = do_manager.email
    spoofed_email = "stranger@openmined.org"

    send_to_inbox(
        ds_manager,
        do_email,
        proposed_message(spoofed_email, do_email, "any.txt", "written by the DS"),
    )
    sync_do(do_manager)

    assert not [n for n in drive_folder_names(do_manager) if spoofed_email in n]
