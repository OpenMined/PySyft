"""E2E test for email auto-approve: reply 'auto-approve' creates an auto-approval object."""

import base64
import json
import tempfile
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

from syft_bg.approve.config import AutoApproveConfig
from syft_bg.approve.orchestrator import ApprovalOrchestrator
from syft_bg.common.config import get_default_paths
from syft_bg.common.state import JsonStateManager
from syft_bg.email_approve.gmail_message import GmailMessage
from syft_bg.email_approve.handler import EmailApproveHandler
from syft_bg.email_approve.monitor import EmailApproveMonitor
from syft_bg.notify.config import NotifyConfig
from syft_bg.notify.gmail.sender import SendResult
from syft_bg.notify.handlers.job import JobHandler
from syft_bg.notify.monitors.job import JobMonitor
from syft_bg.notify.orchestrator import NotificationOrchestrator
from syft_client.sync.syftbox_manager import SyftboxManager

FAKE_THREAD_ID = "thread_auto_approve_123"


@contextmanager
def _temp_config_paths():
    """Redirect config and auto_approvals_dir to a temp directory."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        original = get_default_paths()
        patched = replace(
            original,
            config=tmp_path / "config.yaml",
            auto_approvals_dir=tmp_path / "auto_approvals",
        )
        with (
            patch("syft_bg.common.config.get_default_paths", return_value=patched),
            patch("syft_bg.approve.config.get_default_paths", return_value=patched),
        ):
            yield patched


def _make_notify_orchestrator(
    do_manager: SyftboxManager,
    tmp: Path,
) -> tuple[NotificationOrchestrator, JsonStateManager, MagicMock]:
    """Create a NotificationOrchestrator with a mocked GmailSender."""
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
    reply_text: str = "auto-approve",
) -> tuple[EmailApproveMonitor, JsonStateManager, MagicMock]:
    """Create email approve components with mocked GmailWatcher."""
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
                    "data": base64.urlsafe_b64encode(reply_text.encode()).decode(),
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

    return monitor, email_approve_state, mock_watcher


def _create_project_code_files_with_json_contents(params: dict) -> Path:
    """Create a project dir with main.py and params.json."""
    project_dir = Path(tempfile.mkdtemp(prefix="test_email_auto_approve_"))
    (project_dir / "main.py").write_text(
        """import json
with open("params.json", "r") as f:
    params = json.load(f)
with open("outputs/result.json", "w") as f:
    json.dump({"params": params, "status": "done"}, f)
"""
    )
    (project_dir / "params.json").write_text(json.dumps(params))
    return project_dir


def test_email_auto_approve_creates_object_and_approves_future_jobs():
    """Full flow: auto-approve reply creates approval object, second job is auto-approved."""

    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        use_in_memory_cache=False,
        sync_automatically=False,
    )

    tmp = Path(tempfile.mkdtemp(prefix="test_email_auto_approve_"))

    with _temp_config_paths():
        # -- Step 1: Submit first job with params.json --
        project_dir = _create_project_code_files_with_json_contents({"run": 1})
        ds_manager.submit_python_job(
            user=do_manager.email,
            code_path=str(project_dir),
            job_name="auto_approve_test.job",
            entrypoint="main.py",
        )
        do_manager.sync()

        assert len(do_manager.jobs) == 1
        assert do_manager.jobs[0].status == "pending"
        job_name = do_manager.jobs[0].name

        # -- Step 2: Notify orchestrator sends email --
        notify_orchestrator, notify_state, mock_sender = _make_notify_orchestrator(
            do_manager, tmp
        )
        notify_orchestrator.run_once()

        mock_sender.notify_new_job.assert_called_once()
        assert notify_state.get_thread_id(job_name) == FAKE_THREAD_ID

        # -- Step 3: Simulate "auto-approve" email reply --
        monitor, email_approve_state, mock_watcher = _make_email_approve_orchestrator(
            do_manager, notify_state, tmp, reply_text="auto-approve"
        )

        mock_pubsub_message = MagicMock()
        mock_pubsub_message.data = json.dumps({"historyId": "12345"}).encode("utf-8")

        monitor._on_pubsub_message(mock_pubsub_message)
        history_id, msg = monitor._history_queue.get(timeout=5)
        monitor._process_history(history_id)
        msg.ack()

        # -- Step 4: Verify first job done and DS gets results --
        assert do_manager.jobs[0].status == "done"

        do_manager.sync()
        ds_manager.sync()
        ds_job = [j for j in ds_manager.jobs if j.name == job_name][0]
        assert len(ds_job.output_paths) > 0

        result = json.loads(ds_job.output_paths[0].read_text())
        assert result["params"]["run"] == 1

        # -- Step 5: Verify auto-approve object was created correctly --
        config = AutoApproveConfig.load()
        obj = config.auto_approvals.objects[job_name]
        content_names = {e.relative_path for e in obj.file_contents}
        assert content_names == {"main.py"}
        assert obj.file_paths == ["params.json"]
        assert obj.peers == [ds_manager.email]

        # -- Step 6: Submit second job with same main.py, different params --
        project_dir_2 = _create_project_code_files_with_json_contents({"run": 2})
        ds_manager.submit_python_job(
            user=do_manager.email,
            code_path=str(project_dir_2),
            job_name="auto_approve_test_2.job",
            entrypoint="main.py",
        )
        do_manager.sync()

        second_job = [
            j for j in do_manager.jobs if j.name == "auto_approve_test_2.job"
        ][0]
        assert second_job.status == "pending"

        # -- Step 7: Run approval orchestrator — should auto-approve --
        approve_config = AutoApproveConfig.load()
        approve_config.do_email = do_manager.email
        approve_config.syftbox_root = do_manager.syftbox_folder
        approve_config.approve_state_path = tmp / "approve_state.json"

        orchestrator = ApprovalOrchestrator(client=do_manager, config=approve_config)
        orchestrator.run_once(monitor_type="jobs")

        # -- Step 8: Verify second job auto-approved and DS gets results --
        second_job = [
            j for j in do_manager.jobs if j.name == "auto_approve_test_2.job"
        ][0]
        assert second_job.status == "done"

        do_manager.sync()
        ds_manager.sync()
        ds_job_2 = [j for j in ds_manager.jobs if j.name == "auto_approve_test_2.job"][
            0
        ]
        assert len(ds_job_2.output_paths) > 0

        result_2 = json.loads(ds_job_2.output_paths[0].read_text())
        assert result_2["params"]["run"] == 2
