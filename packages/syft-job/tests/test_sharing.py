"""Run files reach the submitter only when a share grants read on them."""

import json
import warnings
from pathlib import Path

import pytest
import yaml
from syft_job.client import JobClient
from syft_job.config import (
    ALLOW_OUTPUT_PERMISSION_FILES_ENV,
    SyftJobConfig,
    protocol_dir_name,
)
from syft_job.read_permissions import LOGS_WARNING, OUTPUTS
from syft_job.job import JobInfo
from syft_job.job_runner import SyftJobRunner
from syft_job.migrations.registry import JOB_PROTOCOL_VERSION
from syft_job.models import JobState, JobStatus
from syft_job.submission import (
    OWNER_RECORD_FILENAME,
    SUBMISSION_HASH_HEADER,
)
from syft_job.traceback_capture import FRAMES_FILENAME
from syft_permissions.spec.ruleset import PERMISSION_FILE_NAME
from syft_perms import SyftPermContext

# The path segment of the current job protocol, e.g. "v2".
PROTOCOL_DIR = protocol_dir_name(JOB_PROTOCOL_VERSION)

DO_EMAIL = "do@test.org"
DS_EMAIL = "ds@test.org"
LOG_FILES = ("stdout.txt", "stderr.txt", "returncode.txt")
# Read permissions of the DS on each kind of run file.
LOGS = {"stdout.txt": [DS_EMAIL], "stderr.txt": [DS_EMAIL]}
FRAMES = {FRAMES_FILENAME: [DS_EMAIL]}
RETURN_CODE = {"returncode.txt": [DS_EMAIL]}

OK_PY = """\
import os

print("hello from job")
os.makedirs("outputs", exist_ok=True)
with open("outputs/result.txt", "w") as f:
    f.write("done")
"""

EXIT_3_PY = """\
import sys

sys.exit(3)
"""

CRASH_PY = """\
x = 1
raise ValueError("account 88213 holds 4120550")
"""


def do_config_for(tmp_path: Path) -> SyftJobConfig:
    return SyftJobConfig(
        syftbox_folder=tmp_path / "SyftBox", current_user_email=DO_EMAIL
    )


def submit_as_ds(
    tmp_path: Path, requested: dict | None = None, code: str = OK_PY
) -> Path:
    """Submit one job as the DS and return its inbox folder; the DO has not seen it."""
    syftbox = tmp_path / "SyftBox"
    syftbox.mkdir(exist_ok=True)
    code_file = tmp_path / "main.py"
    code_file.write_text(code)
    ds_config = SyftJobConfig(syftbox_folder=syftbox, current_user_email=DS_EMAIL)
    return JobClient(config=ds_config).submit_python_job(
        user=DO_EMAIL,
        code_path=str(code_file),
        job_name="test.job",
        share_files=requested,
    )


def reload_job(tmp_path: Path) -> JobInfo:
    """The job as the DO lists it now."""
    return JobClient(config=do_config_for(tmp_path)).jobs[0]


def submit_only(
    tmp_path: Path, requested: dict | None = None, code: str = OK_PY
) -> JobInfo:
    """Submit one job and return it as the DO sees it, still pending."""
    submit_as_ds(tmp_path, requested, code)
    return reload_job(tmp_path)


def run_approved(tmp_path: Path, share_logs: bool | None = None) -> None:
    """Run the approved jobs as the DO. ``share_logs=None`` uses the default."""
    kwargs = {} if share_logs is None else {"share_logs_with_submitter": share_logs}
    SyftJobRunner(config=do_config_for(tmp_path)).process_approved_jobs(
        stream_output=False, timeout=180, **kwargs
    )


def run_job(
    tmp_path: Path,
    code: str,
    share_logs: bool | None,
    requested: dict | None = None,
    grants: dict | None = None,
):
    """Submit and run one job. Returns (job, review_dir).

    ``requested`` is what the DS asks for, and ``grants`` is what the DO
    approves.
    """
    submit_only(tmp_path, requested, code).approve(read_permissions=grants)
    run_approved(tmp_path, share_logs)
    do_config = do_config_for(tmp_path)
    return (
        JobClient(config=do_config).jobs[0],
        do_config.get_review_job_dir(DO_EMAIL, DS_EMAIL, "test.job"),
    )


def test_state_yaml_omits_exit_code(tmp_path):
    """The DS reads state.yaml, so the exact exit code must not be in it."""
    job, review = run_job(tmp_path, EXIT_3_PY, share_logs=None)
    state = JobState.load(review / "state.yaml")
    assert state.status == JobStatus.FAILED
    assert state.return_code is None
    assert (review / "returncode.txt").read_text().strip() == "3"
    assert not ds_can_read(tmp_path, "returncode.txt")


def test_failure_after_run_writes_return_code(tmp_path, monkeypatch):
    """The runner writes -1 when it fails after the job ran."""

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(SyftJobRunner, "_copy_outputs_to_review", fail)
    job, review = run_job(tmp_path, OK_PY, share_logs=None)

    assert (review / "returncode.txt").read_text().strip() == "-1"
    assert not ds_can_read(tmp_path, "returncode.txt")
    state = JobState.load(review / "state.yaml")
    assert state.status == JobStatus.FAILED
    assert state.return_code is None


def test_share_logs_grants_read(tmp_path):
    job, _ = run_job(tmp_path, OK_PY, share_logs=False)
    job.share_logs([DS_EMAIL])
    for name in LOG_FILES:
        assert ds_can_read(tmp_path, name)


def test_default_holds_logs_back(tmp_path):
    """The DS reads a log only after the DO shares it."""
    job, review = run_job(tmp_path, OK_PY, share_logs=None)
    for name in LOG_FILES:
        assert (review / name).exists()
        assert not ds_can_read(tmp_path, name)
    assert (review / "returncode.txt").read_text().strip() == "0"


def test_share_logs_flag_shares_logs(tmp_path):
    run_job(tmp_path, OK_PY, share_logs=True)
    for name in LOG_FILES:
        assert ds_can_read(tmp_path, name)


def test_second_share_is_harmless(tmp_path):
    job, _ = run_job(tmp_path, OK_PY, share_logs=False)
    job.share_logs([DS_EMAIL])
    job.share_logs([DS_EMAIL])
    assert ds_can_read(tmp_path, "stdout.txt")


def test_owner_reads_unshared_log_through_viewer(tmp_path):
    job, _ = run_job(tmp_path, OK_PY, share_logs=False)
    assert "hello from job" in str(job.stdout)


def test_traceback_record_waits_for_share(tmp_path):
    job, review = run_job(tmp_path, CRASH_PY, share_logs=False)
    record_path = review / FRAMES_FILENAME
    assert record_path.exists()
    assert not ds_can_read(tmp_path, FRAMES_FILENAME)

    record = json.loads(record_path.read_text())
    assert record["chain"][0]["type"] == "ValueError"
    assert "4120550" not in record_path.read_text()

    job.share({FRAMES_FILENAME: [DS_EMAIL]})
    assert ds_can_read(tmp_path, FRAMES_FILENAME)


def test_share_with_nothing_to_share_skips_permissions(tmp_path, monkeypatch):
    """The permission context scans the whole datasite, so build it only to grant."""
    job, _ = run_job(tmp_path, OK_PY, share_logs=None)

    def no_permission_work(self):
        raise AssertionError("share built a permission context")

    monkeypatch.setattr(JobInfo, "_get_perm_context", no_permission_work)

    job.share({FRAMES_FILENAME: [DS_EMAIL]})


def test_rerun_clears_files_of_old_run(tmp_path):
    """A log from the old run must not survive into a later share."""
    job, review = run_job(tmp_path, OK_PY, share_logs=False)
    job.share_outputs([DS_EMAIL])
    assert (review / "stdout.txt").exists()

    job.rerun()

    for name in ("stdout.txt", "stderr.txt", "returncode.txt", "outputs"):
        assert not (review / name).exists()


def test_rerun_drops_log_grant_of_old_run(tmp_path):
    """A log shared in the old run must not reach the DS from the rerun."""
    JobClient(config=do_config_for(tmp_path)).setup_ds_job_folder_as_do(DS_EMAIL)
    job, _ = run_job(tmp_path, OK_PY, share_logs=True)

    job.rerun()
    run_approved(tmp_path, share_logs=False)

    for name in LOG_FILES:
        assert not ds_can_read(tmp_path, name)


def test_rerun_clears_crash_record(tmp_path):
    """A crash record from the old run must not survive a later share.

    A rerun that succeeds writes no record, so a leftover record would describe
    a failure that the new run never had.
    """
    job, review = run_job(tmp_path, CRASH_PY, share_logs=False)
    assert (review / FRAMES_FILENAME).exists()

    job.rerun()

    assert not (review / FRAMES_FILENAME).exists()


def permission_file_code(name: str) -> str:
    """Job code that writes a result and a permission file ``name`` to outputs/."""
    return f"""\
import os

os.makedirs("outputs", exist_ok=True)
with open("outputs/result.txt", "w") as f:
    f.write("done")
with open("outputs/{name}", "w") as f:
    f.write("rules:\\n- pattern: '**'\\n  access:\\n    read: ['*']\\n")
"""


@pytest.mark.parametrize("name", [PERMISSION_FILE_NAME, "Syft.Pub.Yaml"])
def test_permission_file_in_outputs_not_copied(tmp_path, monkeypatch, name):
    """A job must not decide who reads its own outputs."""
    monkeypatch.delenv(ALLOW_OUTPUT_PERMISSION_FILES_ENV, raising=False)
    job, review = run_job(tmp_path, permission_file_code(name), share_logs=None)

    assert job.status == "done"
    assert [path.name for path in (review / "outputs").iterdir()] == ["result.txt"]
    assert not ds_can_read(tmp_path, "outputs/result.txt")


def test_permission_file_in_outputs_kept_when_allowed(tmp_path, monkeypatch):
    monkeypatch.setenv(ALLOW_OUTPUT_PERMISSION_FILES_ENV, "true")
    code = permission_file_code(PERMISSION_FILE_NAME)
    _, review = run_job(tmp_path, code, share_logs=None)

    assert (review / "outputs" / PERMISSION_FILE_NAME).exists()


def test_bad_output_permission_files_value_raises(tmp_path, monkeypatch):
    monkeypatch.setenv(ALLOW_OUTPUT_PERMISSION_FILES_ENV, "yes")
    with pytest.raises(ValueError, match="must be 'true' or 'false', got 'yes'"):
        do_config_for(tmp_path)


def test_symlinked_outputs_copied_as_files(tmp_path):
    external = tmp_path / "external.txt"
    external.write_text("external")
    code = f"""\
import os

os.makedirs("outputs", exist_ok=True)
with open("checkpoint.txt", "w") as f:
    f.write("checkpoint")
os.symlink("../checkpoint.txt", "outputs/latest.txt")
os.symlink({str(external)!r}, "outputs/external.txt")
"""
    _, review = run_job(tmp_path, code, share_logs=None)

    for name, content in (("latest.txt", "checkpoint"), ("external.txt", "external")):
        copied = review / "outputs" / name
        assert not copied.is_symlink()
        assert copied.read_text() == content


def test_symlinked_outputs_folder_copied_as_folder(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    (results / "result.txt").write_text("done")
    code = f"""\
import os
import shutil

shutil.rmtree("outputs", ignore_errors=True)
os.symlink({str(results)!r}, "outputs")
"""
    job, review = run_job(tmp_path, code, share_logs=None)

    assert not (review / "outputs").is_symlink()
    assert (review / "outputs" / "result.txt").read_text() == "done"
    job.rerun()
    assert not (review / "outputs").exists()


def test_rerun_drops_outputs_readers_of_old_run(tmp_path):
    """A reader of the old outputs must not read the outputs of the rerun."""
    peer = "peer@test.org"
    job, _ = run_job(tmp_path, OK_PY, share_logs=None)
    # As on a datasite: the DS review folder has its own permission file, so the
    # outputs grant goes there, not inside outputs/.
    JobClient(config=do_config_for(tmp_path)).setup_ds_job_folder_as_do(DS_EMAIL)
    job.share_outputs([DS_EMAIL, peer])

    job.rerun()
    run_approved(tmp_path)
    reload_job(tmp_path).share_outputs([DS_EMAIL])

    ctx = SyftPermContext(datasite=tmp_path / "SyftBox" / DO_EMAIL)
    result = ctx.open(
        f"app_data/job/review/{DS_EMAIL}/{PROTOCOL_DIR}/test.job/outputs/result.txt"
    )
    assert result.has_read_access(DS_EMAIL)
    assert not result.has_read_access(peer)


# -- read permissions: requested by the DS, approved by the DO ---------------------


def ds_can_read(tmp_path: Path, name: str) -> bool:
    ctx = SyftPermContext(datasite=tmp_path / "SyftBox" / DO_EMAIL)
    return ctx.open(
        f"app_data/job/review/{DS_EMAIL}/{PROTOCOL_DIR}/test.job/{name}"
    ).has_read_access(DS_EMAIL)


def ds_job(tmp_path: Path) -> JobInfo:
    """The job as the DS sees it, on the DO's datasite."""
    ds_config = SyftJobConfig(
        syftbox_folder=tmp_path / "SyftBox", current_user_email=DS_EMAIL
    )
    return JobClient(config=ds_config).jobs["test.job"]


def write_headers(inbox: Path, **headers) -> None:
    """Rewrite config.yaml headers, as the DS can before the DO receives the job."""
    path = inbox / "config.yaml"
    data = yaml.safe_load(path.read_text())
    data["headers"].update(headers)
    path.write_text(yaml.safe_dump(data))


def test_receipt_copies_request_into_state(tmp_path):
    job = submit_only(tmp_path, {**LOGS, **FRAMES})
    assert job.requested_read_permissions == {**LOGS, **FRAMES}
    state = yaml.safe_load((job.job_review_path / "state.yaml").read_text())
    assert state["requested_read_permissions"] == {**LOGS, **FRAMES}
    assert state["approved_read_permissions"] == {}


def test_receipt_drops_request_entries_outside_job(tmp_path):
    inbox = submit_as_ds(tmp_path)
    request = {
        "stdout.txt": [DS_EMAIL],
        "outputs/result.txt": [DS_EMAIL],
        f"../{OWNER_RECORD_FILENAME}": [DS_EMAIL],
        "/etc/passwd": [DS_EMAIL],
        "stderr.txt": [],
    }
    write_headers(inbox, requested_read_permissions=request)
    assert reload_job(tmp_path).requested_read_permissions == {
        "stdout.txt": [DS_EMAIL],
        "outputs/result.txt": [DS_EMAIL],
    }


@pytest.mark.parametrize(
    "request_value, error",
    [
        (["logs"], TypeError),
        ({"stdout.txt": []}, ValueError),
        ({"stdout.txt": DS_EMAIL}, TypeError),
    ],
)
def test_submit_refuses_invalid_request(tmp_path, request_value, error):
    with pytest.raises(error):
        submit_as_ds(tmp_path, request_value)


def submit_with(tmp_path: Path, **share) -> JobInfo:
    """Submit with share_* arguments, and return the job as the DO sees it."""
    (tmp_path / "SyftBox").mkdir(exist_ok=True)
    code_file = tmp_path / "main.py"
    code_file.write_text(OK_PY)
    ds_config = SyftJobConfig(
        syftbox_folder=tmp_path / "SyftBox", current_user_email=DS_EMAIL
    )
    JobClient(config=ds_config).submit_python_job(
        user=DO_EMAIL, code_path=str(code_file), job_name="test.job", **share
    )
    return reload_job(tmp_path)


@pytest.mark.parametrize(
    "share, requested",
    [
        ({"share_logs": True}, LOGS),
        ({"share_outputs": True}, {OUTPUTS: [DS_EMAIL]}),
        ({"share_logs": True, "share_outputs": True}, {**LOGS, OUTPUTS: [DS_EMAIL]}),
        ({}, {}),
    ],
)
def test_share_flags_build_request(tmp_path, share, requested):
    assert submit_with(tmp_path, **share).requested_read_permissions == requested


def test_share_files_with_flag_raises(tmp_path):
    with pytest.raises(ValueError, match="share_files or share_logs"):
        submit_with(tmp_path, share_logs=True, share_files=FRAMES)


def test_approve_share_requested_files_approves_request(tmp_path):
    job = submit_only(tmp_path, {**LOGS, **FRAMES})
    job.approve(share_requested_files=True)
    assert reload_job(tmp_path).approved_read_permissions == {**LOGS, **FRAMES}


def test_approve_with_both_raises(tmp_path):
    job = submit_only(tmp_path, LOGS)
    with pytest.raises(ValueError, match="read_permissions or share_requested_files"):
        job.approve(read_permissions=LOGS, share_requested_files=True)
    assert reload_job(tmp_path).status == "pending"


def test_runner_share_requested_files_shares_request(tmp_path):
    submit_only(tmp_path, {**LOGS, **RETURN_CODE}).approve()
    SyftJobRunner(config=do_config_for(tmp_path)).process_approved_jobs(
        stream_output=False, timeout=180, share_requested_files=True
    )
    assert reload_job(tmp_path).approved_read_permissions == {**LOGS, **RETURN_CODE}
    for name in ("stdout.txt", "stderr.txt", "returncode.txt"):
        assert ds_can_read(tmp_path, name)


def test_approved_outputs_shared_after_run(tmp_path):
    outputs = {OUTPUTS: [DS_EMAIL]}
    run_job(tmp_path, OK_PY, None, requested=outputs, grants=outputs)
    assert ds_can_read(tmp_path, f"{OUTPUTS}/result.txt")


def test_ds_header_cannot_approve(tmp_path):
    """An approval that the DS writes into its submission is ignored."""
    inbox = submit_as_ds(tmp_path, LOGS)
    write_headers(inbox, approved_read_permissions=LOGS)
    job = reload_job(tmp_path)
    assert job.approved_read_permissions == {}

    job.approve()
    run_approved(tmp_path)

    assert not ds_can_read(tmp_path, "stdout.txt")


def test_ds_cannot_approve_or_update(tmp_path):
    submit_only(tmp_path, LOGS)
    with pytest.raises(PermissionError):
        ds_job(tmp_path).approve(read_permissions=LOGS)
    reload_job(tmp_path).approve()
    with pytest.raises(PermissionError):
        ds_job(tmp_path).update_read_permissions(LOGS)
    assert reload_job(tmp_path).approved_read_permissions == {}


def test_ds_cannot_write_state(tmp_path):
    (tmp_path / "SyftBox").mkdir()
    JobClient(config=do_config_for(tmp_path)).setup_ds_job_folder_as_do(DS_EMAIL)
    job = submit_only(tmp_path, LOGS)
    ctx = SyftPermContext(datasite=tmp_path / "SyftBox" / DO_EMAIL)
    state = job.job_review_path / "state.yaml"
    rel = state.relative_to(tmp_path / "SyftBox" / DO_EMAIL)
    assert ctx.open(str(rel)).has_read_access(DS_EMAIL)
    assert not ctx.open(str(rel)).has_write_access(DS_EMAIL)


def test_approval_keeps_requested_files_and_emails_only(tmp_path):
    job = submit_only(tmp_path, {"stdout.txt": [DS_EMAIL, "other@test.org"]})
    job.approve(
        read_permissions={
            "stdout.txt": [DS_EMAIL, "outsider@test.org"],
            "stderr.txt": [DS_EMAIL],
        }
    )
    assert job.approved_read_permissions == {"stdout.txt": [DS_EMAIL]}
    assert reload_job(tmp_path).approved_read_permissions == {"stdout.txt": [DS_EMAIL]}


def test_approved_files_shared_after_run(tmp_path):
    """An approved file that the DS did not request gets no grant."""
    run_job(tmp_path, CRASH_PY, None, requested=LOGS, grants={**LOGS, **FRAMES})
    for name in ("stdout.txt", "stderr.txt"):
        assert ds_can_read(tmp_path, name)
    assert not ds_can_read(tmp_path, FRAMES_FILENAME)
    assert not ds_can_read(tmp_path, "returncode.txt")


def test_request_without_approval_shares_nothing(tmp_path):
    job, _ = run_job(
        tmp_path, CRASH_PY, None, requested={**LOGS, **FRAMES, **RETURN_CODE}
    )
    assert job.approved_read_permissions == {}
    for name in (*LOG_FILES, FRAMES_FILENAME):
        assert not ds_can_read(tmp_path, name)


def test_approved_frames_shared_without_logs(tmp_path):
    run_job(tmp_path, CRASH_PY, None, requested=FRAMES, grants=FRAMES)
    assert ds_can_read(tmp_path, FRAMES_FILENAME)
    assert not ds_can_read(tmp_path, "stderr.txt")


def test_later_approval_shares_finished_job(tmp_path):
    job, _ = run_job(tmp_path, OK_PY, None, requested=RETURN_CODE)
    assert not ds_can_read(tmp_path, "returncode.txt")

    assert job.update_read_permissions(RETURN_CODE) == RETURN_CODE

    assert ds_can_read(tmp_path, "returncode.txt")
    assert not ds_can_read(tmp_path, "stdout.txt")
    assert reload_job(tmp_path).approved_read_permissions == RETURN_CODE


def test_later_approval_through_stale_job_shares(tmp_path):
    """A job object read before the run still sees that the run finished."""
    job = submit_only(tmp_path, LOGS)
    job.approve()
    run_approved(tmp_path)

    job.update_read_permissions(LOGS)

    assert ds_can_read(tmp_path, "stdout.txt")


def test_update_read_permissions_needs_approval_first(tmp_path):
    job = submit_only(tmp_path)
    with pytest.raises(ValueError, match="Approve the job first"):
        job.update_read_permissions(LOGS)


def test_approval_survives_rerun(tmp_path):
    job, _ = run_job(tmp_path, OK_PY, None, requested=LOGS, grants=LOGS)
    job.rerun()
    assert reload_job(tmp_path).approved_read_permissions == LOGS


# -- warning and display --------------------------------------------------------


def logs_warnings(record) -> list:
    return [w for w in record if str(w.message) == LOGS_WARNING]


def test_approve_warns_when_logs_go_out(tmp_path):
    job = submit_only(tmp_path, LOGS)
    with pytest.warns(UserWarning, match="stdout or stderr") as record:
        job.approve(read_permissions=LOGS)
    # The warning points at the caller, not at the helper.
    assert logs_warnings(record)[0].filename == __file__


def test_unrequested_logs_approval_does_not_warn(tmp_path):
    """Approving a file the DS did not request approves nothing."""
    job = submit_only(tmp_path, FRAMES)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        job.approve(read_permissions={**LOGS, **FRAMES})
    assert logs_warnings(record) == []


def test_update_read_permissions_warns_when_logs_go_out(tmp_path):
    job = submit_only(tmp_path, LOGS)
    job.approve()
    with pytest.warns(UserWarning, match="stdout or stderr"):
        job.update_read_permissions(LOGS)


def test_display_shows_read_permissions(tmp_path):
    job = submit_only(tmp_path, {**LOGS, **FRAMES})
    job.approve(read_permissions={**FRAMES, **RETURN_CODE})

    # Only the requested file is kept, so returncode.txt is not approved.
    frames = f"{FRAMES_FILENAME} ({DS_EMAIL})"
    requested = f"stderr.txt ({DS_EMAIL}), stdout.txt ({DS_EMAIL}), {frames}"
    assert job.read_permission_rows() == [
        ("Requested", requested),
        ("Approved", frames),
    ]
    assert f"Approved: {frames}" in str(job)
    assert "Approved:" in job._repr_html_()


def test_display_without_read_permissions(tmp_path):
    job = submit_only(tmp_path, None)
    assert "Requested: none; Approved: none" in str(job)


# -- the request is DS-writable after the approval ------------------------------


def edit_request(job: JobInfo, requested: dict) -> None:
    """Rewrite the request in config.yaml, as a DS can after the approval."""
    write_headers(job.job_submission_path, requested_read_permissions=requested)


def test_request_edit_after_approval_adds_nothing(tmp_path):
    """A wider approval than the request must not share the extra files later."""
    job = submit_only(tmp_path, FRAMES)
    job.approve(read_permissions={**LOGS, **FRAMES})
    edit_request(job, {**LOGS, **FRAMES})

    run_approved(tmp_path)

    assert not ds_can_read(tmp_path, "stdout.txt")


def test_approve_refuses_read_permissions_as_reason(tmp_path):
    """``approve({...})`` would approve nothing and record the dict as reason."""
    job = submit_only(tmp_path, LOGS)
    with pytest.raises(TypeError, match="read_permissions="):
        job.approve(LOGS)
    assert job.status == "pending"


# -- the submission is DS-writable: received complete, approved and run by hash --

ATTACK_RUN_SH = "echo attacker ran\n"


def drop_declared_hash(inbox: Path) -> None:
    """Remove the submission hash from config.yaml, as a DS client before it did."""
    config = inbox / "config.yaml"
    data = yaml.safe_load(config.read_text())
    del data["headers"][SUBMISSION_HASH_HEADER]
    config.write_text(yaml.safe_dump(data))


def test_edit_after_approval_fails_run(tmp_path):
    job = submit_only(tmp_path)
    job.approve()
    (job.job_submission_path / "run.sh").write_text(ATTACK_RUN_SH)

    run_approved(tmp_path)

    job = reload_job(tmp_path)
    assert job.status == "failed"
    assert not (job.job_review_path / "stdout.txt").exists()


def test_approve_refuses_changed_submission(tmp_path):
    job = submit_only(tmp_path)
    (job.job_submission_path / "run.sh").write_text(ATTACK_RUN_SH)

    with pytest.raises(ValueError, match="changed after it was received"):
        job.approve()
    assert reload_job(tmp_path).status == "pending"


@pytest.mark.parametrize("origin", ["deposit_then_rerun", "approved_before_hashes"])
def test_job_without_approved_hash_goes_back_to_pending(tmp_path, origin):
    job = submit_only(tmp_path)
    if origin == "deposit_then_rerun":
        result = tmp_path / "result.txt"
        result.write_text("result")
        job.accept_by_depositing_result(str(result))
        reload_job(tmp_path).rerun()
    else:
        job.approve()
        (job.job_review_path / OWNER_RECORD_FILENAME).unlink()

    run_approved(tmp_path)

    job = reload_job(tmp_path)
    assert job.status == "pending"
    assert not (job.job_review_path / "stdout.txt").exists()


@pytest.mark.parametrize("client", ["declares_hash", "declares_no_hash"])
def test_job_is_received_only_when_complete(tmp_path, client):
    """Sync delivers a job in batches: a job missing a file is not received yet."""
    inbox = submit_as_ds(tmp_path)
    if client == "declares_no_hash":
        drop_declared_hash(inbox)
    main_py = inbox / "code" / "main.py"
    main_py.unlink()

    assert reload_job(tmp_path).status == "received"

    main_py.write_text(OK_PY)
    assert reload_job(tmp_path).status == "pending"


def test_symlink_is_rejected_at_receipt(tmp_path):
    inbox = submit_as_ds(tmp_path)
    drop_declared_hash(inbox)
    secret = tmp_path / "secret.txt"
    secret.write_text("private")
    (inbox / "code" / "link.txt").symlink_to(secret)

    job = reload_job(tmp_path)
    assert job.status == "rejected"
    assert "symlink" in job.review_reason


def test_inbox_is_read_only_for_submitter_after_receipt(tmp_path):
    job = submit_only(tmp_path)
    JobClient(config=do_config_for(tmp_path)).setup_ds_job_folder_as_do(DS_EMAIL)
    datasite = tmp_path / "SyftBox" / DO_EMAIL
    ctx = SyftPermContext(datasite=datasite)
    received = job.job_submission_path.relative_to(datasite)
    not_yet_submitted = received.parent / "next.job"

    assert not ctx.open(f"{received}/run.sh").has_write_access(DS_EMAIL)
    assert ctx.open(f"{received}/run.sh").has_read_access(DS_EMAIL)
    assert ctx.open(f"{not_yet_submitted}/run.sh").has_write_access(DS_EMAIL)


def test_local_copy_error_leaves_job_approved(tmp_path, monkeypatch, caplog):
    """A disk or permission error is not a changed submission: retry next cycle."""
    submit_only(tmp_path).approve()

    def disk_full(source, target):
        raise OSError("disk full")

    monkeypatch.setattr("syft_job.job_runner.copy_submission", disk_full)
    run_approved(tmp_path)

    job = reload_job(tmp_path)
    assert job.status == "approved"
    assert not (job.job_review_path / "stdout.txt").exists()
    assert "disk full" in caplog.text


def test_pending_job_from_before_hashes_gets_record_and_lock(tmp_path):
    job = submit_only(tmp_path)
    (job.job_review_path / OWNER_RECORD_FILENAME).unlink()
    (job.job_submission_path / PERMISSION_FILE_NAME).unlink()

    job = reload_job(tmp_path)

    assert (job.job_review_path / OWNER_RECORD_FILENAME).exists()
    assert (job.job_submission_path / PERMISSION_FILE_NAME).exists()
