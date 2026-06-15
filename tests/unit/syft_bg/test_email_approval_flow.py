"""End-to-end test for email-based job approval flow with mocked connections."""

import base64
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from syft_bg.common.state import JsonStateManager
from syft_bg.email_approve.config import EmailApproveConfig
from syft_bg.email_approve.gmail_message import GmailMessage
from syft_bg.email_approve.handler import EmailApproveHandler
from syft_bg.email_approve.monitor import EmailApproveMonitor
from syft_bg.email_approve.orchestrator import EmailApproveOrchestrator
from syft_bg.notify.config import NotifyConfig
from syft_bg.notify.gmail.sender import SendResult
from syft_bg.notify.handlers.job import JobHandler
from syft_bg.notify.monitors.job import JobMonitor
from syft_bg.notify.orchestrator import NotificationOrchestrator
from syft_client.sync.syftbox_manager import SyftboxManager

from tests.unit.utils import create_test_project_folder, create_tmp_dataset_files

FAKE_THREAD_ID = "thread_abc_123"


def _make_notify_orchestrator(
    do_manager: SyftboxManager,
    tmp: Path,
) -> tuple[NotificationOrchestrator, JsonStateManager, MagicMock]:
    """Create a NotificationOrchestrator with a mocked GmailSender.

    Returns (orchestrator, notify_state, mock_sender).
    """
    notify_state = JsonStateManager(tmp / "notify_state.json")

    mock_sender = MagicMock()
    mock_sender.notify_new_job.return_value = SendResult(
        success=True, thread_id=FAKE_THREAD_ID
    )

    job_handler = JobHandler(
        sender=mock_sender,
        state=notify_state,
        do_email=do_manager.email,
        syftbox_root=Path(do_manager.syftbox_folder),
    )

    job_monitor = JobMonitor(
        syftbox_root=Path(do_manager.syftbox_folder),
        do_email=do_manager.email,
        handler=job_handler,
        state=notify_state,
    )
    notify_config = NotifyConfig(
        do_email=do_manager.email,
        syftbox_root=Path(do_manager.syftbox_folder),
    )
    orchestrator = NotificationOrchestrator(
        config=notify_config,
        job_monitor=job_monitor,
    )
    return orchestrator, notify_state, mock_sender


def _make_email_approve_orchestrator(
    do_manager: SyftboxManager,
    notify_state: JsonStateManager,
    tmp: Path,
) -> tuple[EmailApproveOrchestrator, EmailApproveMonitor, JsonStateManager, MagicMock]:
    """Create an EmailApproveOrchestrator with mocked GmailWatcher and credentials.

    Returns (orchestrator, monitor, email_approve_state, mock_watcher).
    """
    do_email = do_manager.email

    email_approve_state = JsonStateManager(tmp / "email_approve_state.json")
    email_approve_state.set_data("email_approve_last_history_id", "10000")

    handler = EmailApproveHandler(
        job_client=do_manager.job_client,
        job_runner=do_manager.job_runner,
        state=email_approve_state,
        notify_state=notify_state,
        do_email=do_email,
    )

    mock_watcher = MagicMock()
    mock_watcher.list_history_message_ids.return_value = ({"msg_001"}, "12345")
    mock_watcher.get_message.return_value = GmailMessage(
        {
            "threadId": FAKE_THREAD_ID,
            "payload": {
                "mimeType": "text/plain",
                "headers": [{"name": "From", "value": do_email}],
                "body": {
                    "data": base64.urlsafe_b64encode(b"approve").decode(),
                },
            },
        }
    )

    monitor = EmailApproveMonitor(
        watcher=mock_watcher,
        handler=handler,
        state=email_approve_state,
        credentials=MagicMock(),
        subscription_path="projects/fake/subscriptions/fake-sub",
        topic_name="projects/fake/topics/fake-topic",
        do_email=do_email,
    )

    config = EmailApproveConfig(
        do_email=do_email,
        syftbox_root=Path(do_manager.syftbox_folder),
        gmail_token_path=tmp / "fake_gmail_token.json",
        gcp_project_id="fake-project",
        pubsub_topic="projects/fake/topics/fake-topic",
        pubsub_subscription="projects/fake/subscriptions/fake-sub",
        email_approve_state_path=tmp / "email_approve_state.json",
        notify_state_path=tmp / "notify_state.json",
    )

    orchestrator = EmailApproveOrchestrator(
        config=config,
        monitor=monitor,
    )
    return orchestrator, monitor, email_approve_state, mock_watcher


def test_email_approval_e2e():
    """Full flow: DS submits job -> notify detects it -> email approval -> DS gets results."""

    # -- Step 1: Create DO/DS clients --
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        use_in_memory_cache=False,
        sync_automatically=False,
    )
    do_email = do_manager.email
    ds_email = ds_manager.email

    # -- Step 2: Create dataset on DO --
    mock_path, private_path, readme_path = create_tmp_dataset_files()
    do_manager.create_dataset(
        name="my dataset",
        mock_path=mock_path,
        private_path=private_path,
        summary="Test dataset",
        readme_path=readme_path,
        users=[ds_email],
    )
    do_manager.sync()
    ds_manager.sync()

    # -- Step 3: Create notify orchestrator with mocked GmailSender --
    tmp = Path(tempfile.mkdtemp(prefix="test_email_approval_"))
    notify_orchestrator, notify_state, mock_sender = _make_notify_orchestrator(
        do_manager, tmp
    )

    # -- Step 4: Submit job from DS --
    project_dir = create_test_project_folder(with_pyproject=False, multiplier=2)
    ds_manager.submit_python_job(
        user=do_email,
        code_path=str(project_dir),
        job_name="email_approve_test.job",
        entrypoint="main.py",
    )
    do_manager.sync()

    assert len(do_manager.jobs) == 1
    assert do_manager.jobs[0].status == "pending"
    job_name = do_manager.jobs[0].name

    # -- Step 5: Run notify orchestrator once --
    # This scans the local inbox, detects the new job, calls mock sender,
    # and stores the thread_id in notify_state.
    notify_orchestrator.run_once()

    mock_sender.notify_new_job.assert_called_once()
    assert notify_state.get_thread_id(job_name) == FAKE_THREAD_ID

    # -- Step 6: Create email approval orchestrator --
    _, email_approve_monitor, email_approve_state, mock_watcher = (
        _make_email_approve_orchestrator(do_manager, notify_state, tmp)
    )

    # -- Step 7: Simulate Pub/Sub message (no threads) --
    mock_pubsub_message = MagicMock()
    mock_pubsub_message.data = json.dumps({"historyId": "12345"}).encode("utf-8")

    email_approve_monitor._on_pubsub_message(mock_pubsub_message)

    history_id, msg = email_approve_monitor._history_queue.get(timeout=5)
    email_approve_monitor._process_history(history_id)
    msg.ack()

    # -- Step 8: Verify results --
    mock_watcher.list_history_message_ids.assert_called_once_with("10000")
    mock_watcher.get_message.assert_called_once_with("msg_001")

    # DO job should be done
    assert do_manager.jobs[0].status == "done"

    # DS should see output files after sync
    # (In production, the sync service pushes DO changes to Drive periodically.
    # In this test we trigger it manually.)
    do_manager.sync()
    ds_manager.sync()
    ds_job = [j for j in ds_manager.jobs if j.name == job_name][0]
    assert len(ds_job.output_paths) > 0

    output_path = ds_job.output_paths[0]
    result = json.loads(output_path.read_text())
    assert result["multiplier"] == 2
    assert "processed" in result
    assert "original" in result

    # State should be updated
    assert email_approve_state.get_data("email_approve_last_history_id") == "12345"
    assert email_approve_state.was_notified(
        f"email_reply_{FAKE_THREAD_ID}", "processed"
    )
