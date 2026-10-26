"""End-to-end job flow under the current protocol layout."""

from pathlib import Path

import yaml
from syft_job.client import JobClient
from syft_job.config import SyftJobConfig
from syft_job.job_runner import SyftJobRunner
from syft_job.migrations import job_registry
from syft_job.config import protocol_dir_name
from syft_job.migrations.registry import JOB_PROTOCOL_VERSION

# The path segment of the current job protocol, e.g. "v2".
PROTOCOL_DIR = protocol_dir_name(JOB_PROTOCOL_VERSION)

DO_EMAIL = "do@test.org"
DS_EMAIL = "ds@test.org"


def test_current_protocol_roundtrip(tmp_path: Path):
    syftbox = tmp_path / "SyftBox"
    syftbox.mkdir()
    do_config = SyftJobConfig(syftbox_folder=syftbox, current_user_email=DO_EMAIL)
    ds_config = SyftJobConfig(syftbox_folder=syftbox, current_user_email=DS_EMAIL)
    ds_client = JobClient(config=ds_config)
    do_client = JobClient(config=do_config)

    job_dir = ds_client.submit_bash_job(
        user=DO_EMAIL, script="echo hello", job_name="new.job"
    )
    assert job_dir.parent.name == PROTOCOL_DIR
    raw = yaml.safe_load((job_dir / "config.yaml").read_text())
    assert raw["canonical_name"] == "JobSubmissionMetadata"
    assert raw["version"] == job_registry.latest_version("JobSubmissionMetadata")

    job = do_client.jobs[0]
    assert job.name == "new.job" and job.status == "pending"
    job.approve()
    assert (job.job_review_path / "state.yaml").parent.parent.name == PROTOCOL_DIR

    SyftJobRunner(config=do_config).process_approved_jobs(
        stream_output=False, timeout=60
    )
    assert do_client.jobs[0].status == "done"
