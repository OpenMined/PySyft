"""Reading/writing jobs across protocol versions.

Protocol 0 is the last released syft-job (0.1.38): no v<n> path segment and no
canonical_name/version fields in the yaml. The checked-in fixture tree under
fixtures/protocol0_syftbox/ replicates exactly what that release wrote to disk.
"""

import shutil
from pathlib import Path

import pytest
import yaml
from syft_job.client import JobClient
from syft_job.config import SyftJobConfig
from syft_job.job_runner import SyftJobRunner
from syft_job.migrations import job_registry
from syft_job.migrations.registry import JOB_PROTOCOL_VERSION
from syft_job.models import JobSubmissionMetadata
from syft_migration import MigrationError
from syft_perms import SyftPermContext

DO_EMAIL = "do@test.org"
DS_EMAIL = "ds@test.org"

PROTOCOL0_FIXTURE = Path(__file__).parent / "fixtures" / "protocol0_syftbox"


def _syftbox_with_protocol0_jobs(tmp_path: Path) -> Path:
    """A syftbox folder seeded with the protocol-0 fixture tree."""
    syftbox = tmp_path / "SyftBox"
    shutil.copytree(PROTOCOL0_FIXTURE, syftbox)
    return syftbox


def _do_client(syftbox: Path) -> JobClient:
    config = SyftJobConfig(syftbox_folder=syftbox, current_user_email=DO_EMAIL)
    return JobClient(config=config)


def test_lists_protocol0_jobs_upgraded_in_memory(tmp_path: Path):
    syftbox = _syftbox_with_protocol0_jobs(tmp_path)
    do_client = _do_client(syftbox)

    jobs = do_client.jobs
    assert {job.name for job in jobs} == {"legacy.job", "legacy.unscanned"}

    job = next(j for j in jobs if j.name == "legacy.job")
    # Loaded into the latest registered versions, emails derived from the layout.
    latest_meta = job_registry.latest_version("JobSubmissionMetadata")
    latest_state = job_registry.latest_version("JobState")
    assert isinstance(job.job_metadata, JobSubmissionMetadata)
    assert job.job_metadata.version == latest_meta
    assert job._state.version == latest_state
    assert job.submitted_by == DS_EMAIL
    assert job.job_metadata.datasite_email == DO_EMAIL
    assert job.status == "pending"


def test_scan_inbox_receives_protocol0_job_in_old_layout(tmp_path: Path):
    syftbox = _syftbox_with_protocol0_jobs(tmp_path)
    do_client = _do_client(syftbox)

    do_client.scan_inbox()

    # State was written next to the old-layout job: no v<n> segment...
    state_path = (
        syftbox
        / DO_EMAIL
        / "app_data/job/review"
        / DS_EMAIL
        / "legacy.unscanned/state.yaml"
    )
    assert state_path.exists()
    # ...and in the old format, without the identity fields.
    raw = yaml.safe_load(state_path.read_text())
    assert "canonical_name" not in raw and "version" not in raw
    assert raw["status"] == "pending"


def test_approve_writes_back_protocol0_format(tmp_path: Path):
    syftbox = _syftbox_with_protocol0_jobs(tmp_path)
    do_client = _do_client(syftbox)

    job = next(j for j in do_client.jobs if j.name == "legacy.job")
    job.approve(reason="ok")

    state_path = (
        syftbox / DO_EMAIL / "app_data/job/review" / DS_EMAIL / "legacy.job/state.yaml"
    )
    # Byte-exact 0.1.38 output: same dump minus the identity fields.
    expected = job._state.model_dump(mode="json")
    expected.pop("canonical_name")
    expected.pop("version")
    assert state_path.read_text() == yaml.dump(expected, default_flow_style=False)
    # A 0.1.38 reader (plain yaml -> model) understands it.
    assert yaml.safe_load(state_path.read_text())["status"] == "approved"


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
    assert job_dir.parent.name == "v1"
    raw = yaml.safe_load((job_dir / "config.yaml").read_text())
    assert raw["canonical_name"] == "JobSubmissionMetadata"
    assert raw["version"] == job_registry.latest_version("JobSubmissionMetadata")

    job = do_client.jobs[0]
    assert job.name == "new.job" and job.status == "pending"
    job.approve()
    assert (job.job_review_path / "state.yaml").parent.parent.name == "v1"

    SyftJobRunner(config=do_config).process_approved_jobs(
        stream_output=False, timeout=60
    )
    assert do_client.jobs[0].status == "done"


def test_mixed_protocol_listing(tmp_path: Path):
    syftbox = _syftbox_with_protocol0_jobs(tmp_path)
    ds_config = SyftJobConfig(syftbox_folder=syftbox, current_user_email=DS_EMAIL)
    JobClient(config=ds_config).submit_bash_job(
        user=DO_EMAIL, script="echo hi", job_name="new.job"
    )

    jobs = _do_client(syftbox).jobs
    assert {job.name for job in jobs} == {"legacy.job", "legacy.unscanned", "new.job"}


def test_reserved_job_name_rejected(tmp_path: Path):
    syftbox = tmp_path / "SyftBox"
    syftbox.mkdir()
    ds_config = SyftJobConfig(syftbox_folder=syftbox, current_user_email=DS_EMAIL)
    ds_client = JobClient(config=ds_config)

    with pytest.raises(ValueError, match="reserved"):
        ds_client.submit_bash_job(user=DO_EMAIL, script="echo hi", job_name="v2")
    # Names that merely resemble the segment are fine.
    ds_client.submit_bash_job(user=DO_EMAIL, script="echo hi", job_name="v2x")
    ds_client.submit_bash_job(user=DO_EMAIL, script="echo hi", job_name="version1")


def test_submit_to_protocol0_peer_writes_old_layout(tmp_path: Path):
    syftbox = tmp_path / "SyftBox"
    syftbox.mkdir()
    ds_config = SyftJobConfig(syftbox_folder=syftbox, current_user_email=DS_EMAIL)
    protocol0_schema = job_registry.schema_for_protocol_version("0")
    ds_client = JobClient(config=ds_config, peer_schemas={DO_EMAIL: protocol0_schema})

    job_dir = ds_client.submit_bash_job(
        user=DO_EMAIL, script="echo hi", job_name="old-peer.job"
    )

    # No version segment, and the old on-disk format.
    assert job_dir.parent.name == DS_EMAIL
    raw = yaml.safe_load((job_dir / "config.yaml").read_text())
    assert "canonical_name" not in raw and "version" not in raw
    # The DS's own (current) client still lists and reads it back.
    assert ds_client.jobs[0].name == "old-peer.job"


def test_ds_perms_cover_v1_subfolder(tmp_path: Path):
    syftbox = tmp_path / "SyftBox"
    syftbox.mkdir()
    do_config = SyftJobConfig(syftbox_folder=syftbox, current_user_email=DO_EMAIL)
    do_client = JobClient(config=do_config)

    do_client.setup_ds_job_folder_as_do(DS_EMAIL)
    v1_job_dir = do_config.get_job_submission_dir(DO_EMAIL, DS_EMAIL, "some.job")
    v1_job_dir.mkdir(parents=True)

    # The grant on inbox/<ds>/ covers the v1/<job> subfolder.
    ctx = SyftPermContext(datasite=syftbox / DO_EMAIL)
    assert ctx.open(f"app_data/job/inbox/{DS_EMAIL}/v1/some.job/").has_write_access(
        DS_EMAIL
    )
    assert ctx.open(f"app_data/job/review/{DS_EMAIL}/v1/").has_read_access(DS_EMAIL)


def test_protocol_version_for_peer_raises_on_unknown(tmp_path: Path):
    syftbox = tmp_path / "SyftBox"
    syftbox.mkdir()
    ds_config = SyftJobConfig(syftbox_folder=syftbox, current_user_email=DS_EMAIL)
    protocol0_schema = job_registry.schema_for_protocol_version("0")
    manager = JobClient(
        config=ds_config, peer_schemas={DO_EMAIL: protocol0_schema}
    ).manager

    assert manager.protocol_version_for_peer(DO_EMAIL) == "0"
    with pytest.raises(MigrationError):
        manager.protocol_version_for_peer("stranger@test.org")
    # Opting out assumes the current protocol.
    assert (
        manager.protocol_version_for_peer("stranger@test.org", raise_on_unknown=False)
        == JOB_PROTOCOL_VERSION
    )
