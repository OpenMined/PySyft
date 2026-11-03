"""The DS saves each message before it sends it, so a dropped connection or a
stopped kernel does not lose a submission."""

import json

import pytest

from syft.sync.connections.drive.drive_p2p import DriveP2PConnection
from syft.sync.connections.exceptions import ConnectionUnavailableError
from syft.sync.connections.types import MessageKind
from syft.sync.peers.peer import PeerState
from syft.sync.syftbox_manager import SyftboxManager, SyftboxManagerConfig
from syft.sync.sync.spool import SPOOL_ENTRY_VERSION, Spool
from tests.unit.test_sync_manager import path_for_job
from tests.unit.utils import grant_job_inbox_access, mock_message


class _Stopped(Exception):
    """The process stopped."""


def _drop_connection(monkeypatch) -> None:
    def fail(self, recipient, filename, data):
        raise ConnectionUnavailableError("Drive cannot be reached.")

    monkeypatch.setattr(DriveP2PConnection, "watcher_send_raw_bytes_to_inbox", fail)


def _restart(ds_manager) -> SyftboxManager:
    """A new DS manager on the folder and the Drive of ``ds_manager``."""
    config = SyftboxManagerConfig._base_config_for_testing(
        email=ds_manager.email,
        syftbox_folder=ds_manager.syftbox_folder,
        has_ds_role=True,
    )
    return SyftboxManager.from_config(config, ds_manager._connection_router.connections)


def _accepted_contents(do_manager) -> list[str]:
    return [
        str(e.content)
        for msg in do_manager._get_all_accepted_events_do()
        for e in msg.events
    ]


def _pair_with_job_inbox(encryption: bool = False):
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        encryption=encryption
    )
    grant_job_inbox_access(do_manager, ds_manager.email)
    return ds_manager, do_manager, path_for_job(do_manager.email, ds_manager.email)


def test_failed_upload_sent_by_sync_after_restart(monkeypatch):
    ds_manager, do_manager, file_path = _pair_with_job_inbox()
    with monkeypatch.context() as m:
        _drop_connection(m)
        with pytest.raises(ConnectionUnavailableError):
            ds_manager._send_file_change(file_path, "job")

    _restart(ds_manager).sync()
    do_manager.sync()
    assert "job" in _accepted_contents(do_manager)


def test_crash_after_upload_sends_once(monkeypatch):
    ds_manager, do_manager, file_path = _pair_with_job_inbox()

    def stop(self, recipient, msg_id):
        raise _Stopped()

    with monkeypatch.context() as m:
        m.setattr(Spool, "remove", stop)
        with pytest.raises(_Stopped):
            ds_manager._send_file_change(file_path, "job")

    _restart(ds_manager).sync()
    inbox = do_manager._connection_router.connections.p2p_con.receive(
        ds_manager.email, MessageKind.PROPOSED_CHANGES
    )
    assert len(inbox) == 1


def test_edits_saved_offline_apply_in_order(monkeypatch):
    ds_manager, do_manager, file_path = _pair_with_job_inbox()
    with monkeypatch.context() as m:
        _drop_connection(m)
        for content in ["first", "second"]:
            with pytest.raises(ConnectionUnavailableError):
                ds_manager._send_file_change(file_path, content)

    ds_manager.sync()
    do_manager.sync()
    assert "second" in _accepted_contents(do_manager)


def test_saved_message_encrypted_with_peer_key_at_upload(monkeypatch):
    ds_manager, do_manager, file_path = _pair_with_job_inbox(encryption=True)
    ds_manager.load_peers()
    with monkeypatch.context() as m:
        _drop_connection(m)
        with pytest.raises(ConnectionUnavailableError):
            ds_manager._send_file_change(file_path, "job")

    # The DO boots with a new key, and the DS learns it, as for an enclave.
    do_manager._peer_store.generate_keys()
    new_bundle = do_manager._peer_store.get_public_bundle()
    ds_manager._peer_store.set_peer_bundle(do_manager.email, new_bundle)
    ds_manager._connection_router.update_peer_state(
        do_manager.email, PeerState.ACCEPTED.value, public_encryption_bundle=new_bundle
    )

    ds_manager.sync()
    do_manager.sync()
    assert "job" in _accepted_contents(do_manager)


def test_spool_entry_of_later_version_refused(tmp_path):
    spool = Spool(tmp_path)
    spool.add("do@example.com", mock_message())
    entry_path = next(spool.recipient_folder("do@example.com").iterdir())
    later_version = SPOOL_ENTRY_VERSION + 1
    data = json.loads(entry_path.read_text())
    data.update(version=later_version, new_setting=True)
    entry_path.write_text(json.dumps(data))

    with pytest.raises(ValueError, match=f"version {later_version}"):
        spool.pending("do@example.com")
