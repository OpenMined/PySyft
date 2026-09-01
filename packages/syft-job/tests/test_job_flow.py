"""End-to-end unit test for the syft-job package lifecycle."""

import re
import time
from pathlib import Path

import pytest

from syft_job.client import JobClient
from syft_job.config import SyftJobConfig
from syft_job.job_runner import SyftJobRunner
from syft_perms import SyftPermContext
from syft_job.models import JobState


DO_EMAIL = "do@test.org"
DS_EMAIL = "ds@test.org"
PEER_EMAIL = "peer@test.org"
DS2_EMAIL = "ds2@test.org"

MAIN_PY = """\
import os

print("hello from job")
os.makedirs("outputs", exist_ok=True)
with open("outputs/result.txt", "w") as f:
    f.write("done")
"""


def test_full_job_lifecycle(tmp_path: Path):
    syftbox = tmp_path / "SyftBox"
    syftbox.mkdir()

    # Write a trivial main.py for the DS to submit
    code_file = tmp_path / "main.py"
    code_file.write_text(MAIN_PY)

    # Both configs share the same syftbox folder
    do_config = SyftJobConfig(syftbox_folder=syftbox, current_user_email=DO_EMAIL)
    ds_config = SyftJobConfig(syftbox_folder=syftbox, current_user_email=DS_EMAIL)

    ds_client = JobClient(config=ds_config)
    do_client = JobClient(config=do_config)
    do_runner = SyftJobRunner(config=do_config)

    # --- DS submits a python job to DO ---
    job_dir = ds_client.submit_python_job(
        user=DO_EMAIL,
        code_path=str(code_file),
        job_name="test.job",
    )
    assert job_dir.exists()
    assert (job_dir / "run.sh").exists()
    assert (job_dir / "config.yaml").exists()
    assert (job_dir / "code" / "main.py").exists()

    # Job dir should be under inbox/<ds_email>/v1/<job_name>
    expected_parent = ds_config.get_all_submissions_dir(DO_EMAIL) / DS_EMAIL / "v1"
    assert job_dir.parent == expected_parent

    # --- DO lists jobs (auto-scans inbox) and sees it as pending ---
    review_path = do_config.get_review_job_dir(DO_EMAIL, DS_EMAIL, "test.job")
    jobs = do_client.jobs

    # Verify state.yaml was auto-created in review/
    assert (review_path / "state.yaml").exists()
    assert len(jobs) == 1
    job = jobs[0]
    assert job.name == "test.job"
    assert job.status == "pending"
    assert job.submitted_by == DS_EMAIL

    # --- DO approves the job ---
    job.approve()
    assert job.status == "approved"

    # --- DO runs approved jobs ---
    do_runner.process_approved_jobs(stream_output=False, timeout=120)

    # Re-fetch to get updated status
    job = do_client.jobs[0]
    assert job.status == "done"

    # --- Check outputs ---
    # output_paths includes syft.pub.yaml created by _prepare_outputs_dir
    output_names = {p.name for p in job.output_paths}
    assert "result.txt" in output_names
    result_file = next(p for p in job.output_paths if p.name == "result.txt")
    assert result_file.read_text().strip() == "done"

    # --- Check stdout / stderr (now in review/) ---
    stdout_path = review_path / "stdout.txt"
    stderr_path = review_path / "stderr.txt"
    assert stdout_path.exists()
    assert stderr_path.exists()
    assert "hello from job" in stdout_path.read_text()

    # --- Check returncode (now in review/) ---
    returncode_path = review_path / "returncode.txt"
    assert returncode_path.exists()
    assert returncode_path.read_text().strip() == "0"

    # --- Before sharing, DS should NOT have read access ---
    ctx = SyftPermContext(datasite=syftbox / DO_EMAIL)
    assert not ctx.open(
        f"app_data/job/review/{DS_EMAIL}/v1/test.job/outputs/"
    ).has_read_access(DS_EMAIL)
    assert not ctx.open(
        f"app_data/job/review/{DS_EMAIL}/v1/test.job/stdout.txt"
    ).has_read_access(DS_EMAIL)
    assert not ctx.open(
        f"app_data/job/review/{DS_EMAIL}/v1/test.job/stderr.txt"
    ).has_read_access(DS_EMAIL)
    assert not ctx.open(
        f"app_data/job/review/{DS_EMAIL}/v1/test.job/returncode.txt"
    ).has_read_access(DS_EMAIL)

    # --- Share outputs and logs with DS ---
    job.share_outputs([DS_EMAIL])
    job.share_logs([DS_EMAIL])

    # --- Verify DS has read access via SyftPermContext ---
    ctx = SyftPermContext(datasite=syftbox / DO_EMAIL)

    outputs_folder = ctx.open(f"app_data/job/review/{DS_EMAIL}/v1/test.job/outputs/")
    assert outputs_folder.has_read_access(DS_EMAIL)

    stdout_file = ctx.open(f"app_data/job/review/{DS_EMAIL}/v1/test.job/stdout.txt")
    assert stdout_file.has_read_access(DS_EMAIL)

    stderr_file = ctx.open(f"app_data/job/review/{DS_EMAIL}/v1/test.job/stderr.txt")
    assert stderr_file.has_read_access(DS_EMAIL)

    returncode_file = ctx.open(
        f"app_data/job/review/{DS_EMAIL}/v1/test.job/returncode.txt"
    )
    assert returncode_file.has_read_access(DS_EMAIL)


def test_ds_job_folder_permissions(tmp_path: Path):
    """Test that setup_ds_job_folder_as_do creates inbox and review folders with correct permissions."""
    syftbox = tmp_path / "SyftBox"
    syftbox.mkdir()

    do_config = SyftJobConfig(syftbox_folder=syftbox, current_user_email=DO_EMAIL)
    do_client = JobClient(config=do_config)

    # Create DS job folder with permissions
    ds_inbox_folder = do_client.setup_ds_job_folder_as_do(DS_EMAIL)
    assert ds_inbox_folder.exists()
    assert ds_inbox_folder == do_config.get_all_submissions_dir(DO_EMAIL) / DS_EMAIL

    # Review folder should also exist
    ds_review_folder = do_config.get_review_dir(DO_EMAIL) / DS_EMAIL
    assert ds_review_folder.exists()

    # DS should have write access to their inbox folder
    ctx = SyftPermContext(datasite=syftbox / DO_EMAIL)
    inbox_folder = ctx.open(f"app_data/job/inbox/{DS_EMAIL}/")
    assert inbox_folder.has_write_access(DS_EMAIL)
    assert inbox_folder.has_read_access(DO_EMAIL)

    # DS should have read access to their review folder
    review_folder = ctx.open(f"app_data/job/review/{DS_EMAIL}/")
    assert review_folder.has_read_access(DS_EMAIL)

    # Another user should NOT have write access
    other_email = "other@test.org"
    assert not inbox_folder.has_write_access(other_email)
    assert not inbox_folder.has_read_access(other_email)


def test_job_reject(tmp_path: Path):
    """Test that a DO can reject a pending job."""
    syftbox = tmp_path / "SyftBox"
    syftbox.mkdir()

    code_file = tmp_path / "main.py"
    code_file.write_text("print('hello')")

    do_config = SyftJobConfig(syftbox_folder=syftbox, current_user_email=DO_EMAIL)
    ds_config = SyftJobConfig(syftbox_folder=syftbox, current_user_email=DS_EMAIL)

    ds_client = JobClient(config=ds_config)
    do_client = JobClient(config=do_config)

    ds_client.submit_python_job(
        user=DO_EMAIL, code_path=str(code_file), job_name="reject.job"
    )
    job = do_client.jobs[0]  # auto-scans inbox
    assert job.status == "pending"

    job.reject(reason="Not approved")
    assert job.status == "rejected"
    assert job.review_reason == "Not approved"


def test_submission_validation(tmp_path: Path):
    """Test that invalid submissions are auto-rejected during scan_inbox."""
    syftbox = tmp_path / "SyftBox"
    syftbox.mkdir()

    do_config = SyftJobConfig(syftbox_folder=syftbox, current_user_email=DO_EMAIL)
    do_client = JobClient(config=do_config)

    # Manually create an invalid submission (missing code/ directory)
    submission_path = do_config.get_job_submission_dir(DO_EMAIL, DS_EMAIL, "bad.job")
    submission_path.mkdir(parents=True)
    (submission_path / "config.yaml").write_text(
        "name: bad.job\ntype: python\nsubmitted_by: ds@test.org\nsubmitted_at: '2025-01-01T00:00:00+00:00'\n"
    )
    (submission_path / "run.sh").write_text("#!/bin/bash\necho hi")
    # Missing code/ directory — should fail validation

    do_client.scan_inbox()

    review_state = (
        do_config.get_review_job_dir(DO_EMAIL, DS_EMAIL, "bad.job") / "state.yaml"
    )
    assert review_state.exists()

    state = JobState.load(review_state)
    assert state.status.value == "rejected"
    assert state.review_reason is not None


INFINITE_LOOP_PY = """\
while True:
    pass
"""


def test_timeout_does_not_hang_runner(tmp_path: Path):
    syftbox = tmp_path / "SyftBox"
    syftbox.mkdir()

    code_file = tmp_path / "main.py"
    code_file.write_text(INFINITE_LOOP_PY)

    do_config = SyftJobConfig(syftbox_folder=syftbox, current_user_email=DO_EMAIL)
    ds_config = SyftJobConfig(syftbox_folder=syftbox, current_user_email=DS_EMAIL)

    ds_client = JobClient(config=ds_config)
    do_client = JobClient(config=do_config)
    do_runner = SyftJobRunner(config=do_config)

    ds_client.submit_python_job(
        user=DO_EMAIL, code_path=str(code_file), job_name="loop.job"
    )
    do_client.jobs[0].approve()

    start = time.time()
    do_runner.process_approved_jobs(stream_output=True, timeout=3)
    elapsed = time.time() - start

    # 3s job timeout + venv setup + tree-kill cleanup should fit well under 60s.
    assert elapsed < 60, f"process_approved_jobs took {elapsed:.1f}s — likely hung"
    assert do_client.jobs[0].status == "failed"


def test_jobs_table_hint_uses_name_based_indexing(tmp_path: Path):
    """The DO hint must point at jobs["name"], not jobs[0].

    Positional indexing is not safe to recommend: the row numbers rendered in the
    table are assigned in per-owner display order, while __getitem__ subscripts
    the underlying time-sorted list, so the two disagree once more than one
    datasite owner has jobs.
    """
    syftbox = tmp_path / "SyftBox"
    syftbox.mkdir()
    code_file = tmp_path / "main.py"
    code_file.write_text(MAIN_PY)

    # has_do_role gates the hint — it is only shown to a data owner.
    do_config = SyftJobConfig(
        syftbox_folder=syftbox, current_user_email=DO_EMAIL, has_do_role=True
    )
    ds_config = SyftJobConfig(syftbox_folder=syftbox, current_user_email=DS_EMAIL)
    ds_client = JobClient(config=ds_config)
    do_client = JobClient(config=do_config)

    ds_client.submit_python_job(
        user=DO_EMAIL, code_path=str(code_file), job_name="analysis.job"
    )

    jobs = do_client.jobs
    text = str(jobs)
    html = jobs._repr_html_()

    for rendering in (text, html):
        assert 'jobs["analysis.job"].approve()' in rendering
        assert "jobs[0]" not in rendering

    # The hint names a job the DO can actually approve.
    assert do_client.jobs["analysis.job"].status == "pending"


def _index_name_pairs_from_text(text: str) -> list[tuple[int, str]]:
    return [(int(i), name) for i, name in re.findall(r"\[(\d+)\s*\]\s+(\S+)", text)]


def _index_name_pairs_from_html(html: str) -> list[tuple[int, str]]:
    return [
        (int(i), name.strip())
        for i, name in re.findall(
            r'class="syftjob-index">\[(\d+)\]</span>.*?'
            r'class="syftjob-td syftjob-job-name">\s*([^<]+)',
            html,
            flags=re.DOTALL,
        )
    ]


def test_jobs_table_index_matches_getitem(tmp_path: Path):
    """The [N] printed in the table must be the N that jobs[N] returns.

    The table groups by datasite owner (root first, then peers). __getitem__
    used to subscript a newest-first list, so with two owners the row labelled
    [0] was not jobs[0].
    """
    syftbox = tmp_path / "SyftBox"
    syftbox.mkdir()
    code_file = tmp_path / "main.py"
    code_file.write_text(MAIN_PY)

    do_config = SyftJobConfig(
        syftbox_folder=syftbox, current_user_email=DO_EMAIL, has_do_role=True
    )
    ds_config = SyftJobConfig(syftbox_folder=syftbox, current_user_email=DS_EMAIL)
    ds_client = JobClient(config=ds_config)
    do_client = JobClient(config=do_config)

    # Older job on the root datasite, then a newer one on a peer datasite.
    ds_client.submit_python_job(
        user=DO_EMAIL, code_path=str(code_file), job_name="older-root-job"
    )
    ds_client.submit_python_job(
        user=PEER_EMAIL, code_path=str(code_file), job_name="newest-peer-job"
    )

    jobs = do_client.jobs
    assert [job.name for job in jobs] == ["older-root-job", "newest-peer-job"]

    text_pairs = _index_name_pairs_from_text(str(jobs))
    html_pairs = _index_name_pairs_from_html(jobs._repr_html_())
    assert text_pairs == [(0, "older-root-job"), (1, "newest-peer-job")]
    assert html_pairs == [(0, "older-root-job"), (1, "newest-peer-job")]

    for index, name in text_pairs + html_pairs:
        assert jobs[index].name == name


def test_jobs_table_hint_skips_an_ambiguous_job_name(tmp_path: Path):
    """The hint must not name a job that two submitters both use.

    Job names are unique per datasite and submitter, so two data scientists can
    submit "analysis" to the same data owner. jobs["analysis"] then raises, so a
    hint that named it would send the data owner straight to an error.
    """
    syftbox = tmp_path / "SyftBox"
    syftbox.mkdir()
    code_file = tmp_path / "main.py"
    code_file.write_text(MAIN_PY)

    do_config = SyftJobConfig(
        syftbox_folder=syftbox, current_user_email=DO_EMAIL, has_do_role=True
    )
    do_client = JobClient(config=do_config)
    ds1_client = JobClient(
        config=SyftJobConfig(syftbox_folder=syftbox, current_user_email=DS_EMAIL)
    )
    ds2_client = JobClient(
        config=SyftJobConfig(syftbox_folder=syftbox, current_user_email=DS2_EMAIL)
    )

    # "solo" first, so the ambiguous name is the newest and would otherwise win.
    ds1_client.submit_python_job(
        user=DO_EMAIL, code_path=str(code_file), job_name="solo"
    )
    ds1_client.submit_python_job(
        user=DO_EMAIL, code_path=str(code_file), job_name="analysis"
    )
    ds2_client.submit_python_job(
        user=DO_EMAIL, code_path=str(code_file), job_name="analysis"
    )

    jobs = do_client.jobs
    for rendering in (str(jobs), jobs._repr_html_()):
        assert 'jobs["solo"].approve()' in rendering
        assert 'jobs["analysis"]' not in rendering

    assert jobs["solo"].name == "solo"
    with pytest.raises(ValueError, match="Multiple jobs are named 'analysis'"):
        jobs["analysis"]


def test_jobs_table_hint_falls_back_to_the_index(tmp_path: Path):
    """With no unambiguous name left, the hint must give a position.

    Every job in the table shares its name here, so jobs["analysis"] raises.
    The hint still has to name something the data owner can run.
    """
    syftbox = tmp_path / "SyftBox"
    syftbox.mkdir()
    code_file = tmp_path / "main.py"
    code_file.write_text(MAIN_PY)

    do_config = SyftJobConfig(
        syftbox_folder=syftbox, current_user_email=DO_EMAIL, has_do_role=True
    )
    do_client = JobClient(config=do_config)
    for ds_email in (DS_EMAIL, DS2_EMAIL):
        JobClient(
            config=SyftJobConfig(syftbox_folder=syftbox, current_user_email=ds_email)
        ).submit_python_job(
            user=DO_EMAIL, code_path=str(code_file), job_name="analysis"
        )

    jobs = do_client.jobs
    for rendering in (str(jobs), jobs._repr_html_()):
        assert "jobs[0].approve()" in rendering
        assert 'jobs["analysis"]' not in rendering

    assert jobs[0].name == "analysis"


def test_skipping_one_job_spares_its_same_named_sibling(tmp_path: Path):
    """A skip must name the submitter as well as the job.

    Two data scientists can submit the same job name to one data owner. Given
    only the name, the runner dropped both — so a job its peer was compatible
    with never ran, and nothing said so.
    """
    syftbox = tmp_path / "SyftBox"
    syftbox.mkdir()
    code_file = tmp_path / "main.py"
    code_file.write_text(MAIN_PY)

    do_config = SyftJobConfig(syftbox_folder=syftbox, current_user_email=DO_EMAIL)
    do_client = JobClient(config=do_config)
    do_runner = SyftJobRunner(config=do_config)

    for ds_email in (DS_EMAIL, DS2_EMAIL):
        JobClient(
            config=SyftJobConfig(syftbox_folder=syftbox, current_user_email=ds_email)
        ).submit_python_job(
            user=DO_EMAIL, code_path=str(code_file), job_name="shared.job"
        )

    for job in do_client.jobs:
        job.approve()

    do_runner.process_approved_jobs(
        stream_output=False, timeout=60, skip_jobs=[("shared.job", DS_EMAIL)]
    )

    final = {job.submitted_by: job.status for job in do_client.jobs}
    assert final[DS_EMAIL] == "approved", "the skipped job must not run"
    assert final[DS2_EMAIL] == "done", "its same-named sibling must still run"


def test_no_hint_when_the_do_owns_none_of_the_jobs(tmp_path: Path):
    """Every subscript would name a job on another datasite, which approve() refuses.

    A hint is worse than no hint when the only command it can give raises
    PermissionError.
    """
    syftbox = tmp_path / "SyftBox"
    syftbox.mkdir()
    code_file = tmp_path / "main.py"
    code_file.write_text(MAIN_PY)

    do_config = SyftJobConfig(
        syftbox_folder=syftbox, current_user_email=DO_EMAIL, has_do_role=True
    )
    do_client = JobClient(config=do_config)
    JobClient(
        config=SyftJobConfig(syftbox_folder=syftbox, current_user_email=DS_EMAIL)
    ).submit_python_job(
        user=PEER_EMAIL, code_path=str(code_file), job_name="peer-owned.job"
    )

    jobs = do_client.jobs
    assert [job.datasite_owner_email for job in jobs] == [PEER_EMAIL]
    for rendering in (str(jobs), jobs._repr_html_()):
        assert "peer-owned.job" in rendering
        assert "💡" not in rendering


def test_runner_ignores_approved_jobs_on_another_datasite(tmp_path: Path):
    """`jobs` spans every datasite in the folder; the runner runs only its own.

    A peer's approved job is visible but was never a candidate, which is why the
    skip report in syft-rds filters on the datasite owner before naming a job as
    one that did not run.
    """
    syftbox = tmp_path / "SyftBox"
    syftbox.mkdir()
    code_file = tmp_path / "main.py"
    code_file.write_text(MAIN_PY)

    do_config = SyftJobConfig(syftbox_folder=syftbox, current_user_email=DO_EMAIL)
    peer_config = SyftJobConfig(syftbox_folder=syftbox, current_user_email=PEER_EMAIL)
    do_client = JobClient(config=do_config)
    peer_client = JobClient(config=peer_config)
    do_runner = SyftJobRunner(config=do_config)

    ds_client = JobClient(
        config=SyftJobConfig(syftbox_folder=syftbox, current_user_email=DS_EMAIL)
    )
    ds_client.submit_python_job(
        user=DO_EMAIL, code_path=str(code_file), job_name="mine.job"
    )
    ds_client.submit_python_job(
        user=PEER_EMAIL, code_path=str(code_file), job_name="theirs.job"
    )

    do_client.jobs["mine.job"].approve()
    peer_client.jobs["theirs.job"].approve()

    # The DO sees both, including the peer's approved job.
    visible = {job.name: job.status for job in do_client.jobs}
    assert visible == {"mine.job": "approved", "theirs.job": "approved"}

    do_runner.process_approved_jobs(stream_output=False, timeout=60)

    after = {job.name: job.status for job in do_client.jobs}
    assert after["mine.job"] == "done"
    assert after["theirs.job"] == "approved", "a peer's job is not this runner's to run"


def test_deprecated_skip_job_names_still_skips_by_name(tmp_path: Path):
    """The old name-only argument keeps its old behaviour, and says it is old.

    It drops every submitter's job of that name — which is the bug — so it warns
    rather than silently changing what an existing caller gets.
    """
    syftbox = tmp_path / "SyftBox"
    syftbox.mkdir()
    code_file = tmp_path / "main.py"
    code_file.write_text(MAIN_PY)

    do_config = SyftJobConfig(syftbox_folder=syftbox, current_user_email=DO_EMAIL)
    do_client = JobClient(config=do_config)
    do_runner = SyftJobRunner(config=do_config)

    for ds_email in (DS_EMAIL, DS2_EMAIL):
        JobClient(
            config=SyftJobConfig(syftbox_folder=syftbox, current_user_email=ds_email)
        ).submit_python_job(
            user=DO_EMAIL, code_path=str(code_file), job_name="shared.job"
        )
    for job in do_client.jobs:
        job.approve()

    with pytest.warns(DeprecationWarning, match="Skipping jobs by name alone"):
        do_runner.process_approved_jobs(
            stream_output=False, timeout=60, skip_job_names=["shared.job"]
        )

    assert [job.status for job in do_client.jobs] == ["approved", "approved"]


def test_approval_error_names_both_parties(tmp_path: Path):
    """The old message called the peer "the admin user" and never named you."""
    syftbox = tmp_path / "SyftBox"
    syftbox.mkdir()
    code_file = tmp_path / "main.py"
    code_file.write_text(MAIN_PY)

    do_client = JobClient(
        config=SyftJobConfig(syftbox_folder=syftbox, current_user_email=DO_EMAIL)
    )
    JobClient(
        config=SyftJobConfig(syftbox_folder=syftbox, current_user_email=DS_EMAIL)
    ).submit_python_job(
        user=PEER_EMAIL, code_path=str(code_file), job_name="peer-owned.job"
    )
    # The owner scans it into pending, so approve() reaches the ownership check.
    JobClient(
        config=SyftJobConfig(syftbox_folder=syftbox, current_user_email=PEER_EMAIL)
    ).scan_inbox()

    job = do_client.jobs["peer-owned.job"]
    assert job.status == "pending"
    with pytest.raises(PermissionError) as exc:
        job.approve()

    message = str(exc.value)
    assert DO_EMAIL in message, "must say who you are"
    assert PEER_EMAIL in message, "must say whose datasite it is"
    assert "peer-owned.job" in message
    assert "admin user" not in message


def test_job_files_do_not_hide_a_non_filesystem_error(tmp_path: Path, monkeypatch):
    """Only the filesystem may truncate the file list; other errors propagate."""
    syftbox = tmp_path / "SyftBox"
    syftbox.mkdir()
    code_file = tmp_path / "main.py"
    code_file.write_text(MAIN_PY)

    do_client = JobClient(
        config=SyftJobConfig(syftbox_folder=syftbox, current_user_email=DO_EMAIL)
    )
    JobClient(
        config=SyftJobConfig(syftbox_folder=syftbox, current_user_email=DS_EMAIL)
    ).submit_python_job(user=DO_EMAIL, code_path=str(code_file), job_name="files.job")

    job = do_client.jobs["files.job"]
    assert job.files, "sanity: the job has files"

    def boom(*args, **kwargs):
        raise RuntimeError("not a filesystem problem")

    monkeypatch.setattr(Path, "rglob", boom)
    with pytest.raises(RuntimeError, match="not a filesystem problem"):
        job.files
