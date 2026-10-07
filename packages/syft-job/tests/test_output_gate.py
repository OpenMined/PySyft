"""The DS reads only the files in review/<ds>/ that the DO released."""

from pathlib import Path

import pytest
from syft_perms import SyftPermContext

from syft_job.client import JobClient
from syft_job.config import SyftJobConfig
from syft_job.disclosures import DISCLOSURES_FILENAME
from syft_job.job_runner import SyftJobRunner

DO_EMAIL = "do@test.org"
DS_EMAIL = "ds@test.org"
JOB_NAME = "gate.job"
REVIEW_REL = f"app_data/job/review/{DS_EMAIL}"
JOB_REVIEW_REL = f"{REVIEW_REL}/v1/{JOB_NAME}"


@pytest.fixture
def env(tmp_path: Path):
    syftbox = tmp_path / "SyftBox"
    syftbox.mkdir()
    do_config = SyftJobConfig(syftbox_folder=syftbox, current_user_email=DO_EMAIL)
    ds_config = SyftJobConfig(syftbox_folder=syftbox, current_user_email=DS_EMAIL)
    return {
        "syftbox": syftbox,
        "do_config": do_config,
        "do_client": JobClient(config=do_config),
        "ds_client": JobClient(config=ds_config),
        "runner": SyftJobRunner(config=do_config),
    }


def _ds_reads(env, rel_path: str) -> bool:
    ctx = SyftPermContext(datasite=env["syftbox"] / DO_EMAIL)
    return ctx.open(rel_path).has_read_access(DS_EMAIL)


def _grant_old_folder_read(env) -> None:
    """Write the grant that earlier versions gave the DS on review/<ds>/."""
    ctx = SyftPermContext(datasite=env["syftbox"] / DO_EMAIL)
    ctx.open(f"{REVIEW_REL}/").grant_read_access(DS_EMAIL)


def _submit_and_approve(env, script: str) -> None:
    env["ds_client"].submit_bash_job(DO_EMAIL, script, JOB_NAME)
    job = env["do_client"].jobs[0]
    job.approve()


def _exfil_script(env) -> str:
    """A job that writes a file beside outputs/, in its own review directory."""
    review_dir = env["do_config"].get_review_job_dir(DO_EMAIL, DS_EMAIL, JOB_NAME)
    return f"echo private > '{review_dir}/exfil.txt'\n"


def test_file_written_beside_outputs_is_not_readable_by_ds(env):
    env["do_client"].setup_ds_job_folder_as_do(DS_EMAIL)
    _submit_and_approve(env, _exfil_script(env))

    env["runner"].process_approved_jobs(
        stream_output=False, share_outputs_with_submitter=True
    )

    review_dir = env["do_config"].get_review_job_dir(DO_EMAIL, DS_EMAIL, JOB_NAME)
    assert (review_dir / "exfil.txt").exists()
    assert not _ds_reads(env, f"{JOB_REVIEW_REL}/exfil.txt")


def test_nested_files_named_like_state_are_not_readable_by_ds(env):
    env["do_client"].setup_ds_job_folder_as_do(DS_EMAIL)
    review_dir = env["do_config"].get_review_job_dir(DO_EMAIL, DS_EMAIL, JOB_NAME)
    script = (
        f"mkdir -p '{review_dir}/nested'\n"
        f"echo private > '{review_dir}/nested/state.yaml'\n"
        f"echo private > '{review_dir}/nested/{DISCLOSURES_FILENAME}'\n"
    )
    _submit_and_approve(env, script)

    env["runner"].process_approved_jobs(stream_output=False)

    assert (review_dir / "nested" / "state.yaml").exists()
    assert not _ds_reads(env, f"{JOB_REVIEW_REL}/nested/state.yaml")
    assert not _ds_reads(env, f"{JOB_REVIEW_REL}/nested/{DISCLOSURES_FILENAME}")
    assert not _ds_reads(env, f"{REVIEW_REL}/{JOB_NAME}/nested/state.yaml")
    assert _ds_reads(env, f"{JOB_REVIEW_REL}/state.yaml")


def test_ds_reads_state_and_disclosures_before_release(env):
    env["do_client"].setup_ds_job_folder_as_do(DS_EMAIL)
    _submit_and_approve(env, "echo hi\n")

    assert _ds_reads(env, f"{JOB_REVIEW_REL}/state.yaml")
    assert _ds_reads(env, f"{JOB_REVIEW_REL}/{DISCLOSURES_FILENAME}")
    assert not _ds_reads(env, f"{JOB_REVIEW_REL}/other.txt")


def test_v0_layout_state_is_readable_by_ds(env):
    env["do_client"].setup_ds_job_folder_as_do(DS_EMAIL)

    assert _ds_reads(env, f"{REVIEW_REL}/{JOB_NAME}/state.yaml")
    assert not _ds_reads(env, f"{REVIEW_REL}/{JOB_NAME}/stdout.txt")


def test_setup_removes_old_folder_grant(env):
    _grant_old_folder_read(env)
    assert _ds_reads(env, f"{JOB_REVIEW_REL}/exfil.txt")

    env["do_client"].setup_ds_job_folder_as_do(DS_EMAIL)

    assert not _ds_reads(env, f"{JOB_REVIEW_REL}/exfil.txt")
    assert _ds_reads(env, f"{JOB_REVIEW_REL}/state.yaml")


def test_setup_twice_keeps_one_rule_per_pattern(env):
    env["do_client"].setup_ds_job_folder_as_do(DS_EMAIL)
    env["do_client"].setup_ds_job_folder_as_do(DS_EMAIL)

    from syft_permissions import PERMISSION_FILE_NAME, RuleSet

    review_dir = env["do_config"].get_review_dir(DO_EMAIL) / DS_EMAIL
    rules = RuleSet.load(review_dir / PERMISSION_FILE_NAME).rules
    patterns = [r.pattern for r in rules]
    assert len(patterns) == len(set(patterns))
    for rule in rules:
        assert rule.access.read.count(DS_EMAIL) <= 1


def test_runner_removes_old_folder_grant_before_the_job_runs(env):
    _grant_old_folder_read(env)
    _submit_and_approve(env, _exfil_script(env))

    env["runner"].process_approved_jobs(stream_output=False)

    assert not _ds_reads(env, f"{JOB_REVIEW_REL}/exfil.txt")
    assert _ds_reads(env, f"{JOB_REVIEW_REL}/state.yaml")


def test_old_grant_removal_keeps_released_files_readable(env):
    """A release under the old grant wrote per-file rules, which stay."""
    _grant_old_folder_read(env)
    _submit_and_approve(env, "echo hi\n")
    env["runner"].process_approved_jobs(
        stream_output=False,
        share_outputs_with_submitter=True,
        share_logs_with_submitter=True,
    )

    env["do_client"].setup_ds_job_folder_as_do(DS_EMAIL)

    assert _ds_reads(env, f"{JOB_REVIEW_REL}/outputs/")
    assert _ds_reads(env, f"{JOB_REVIEW_REL}/stdout.txt")
