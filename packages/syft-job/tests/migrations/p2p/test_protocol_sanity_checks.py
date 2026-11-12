"""Sanity checks around the protocol-versioned job layout."""

from pathlib import Path

import pytest
from syft_job.client import JobClient
from syft_job.config import SyftJobConfig, protocol_dir_name
from syft_job.migrations import job_registry
from syft_job.migrations.registry import JOB_PROTOCOL_VERSION
from syft_migration import MigrationError
from syft_perms import SyftPermContext

DO_EMAIL = "do@test.org"
DS_EMAIL = "ds@test.org"


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


def test_ds_perms_cover_the_protocol_subfolder(tmp_path: Path):
    syftbox = tmp_path / "SyftBox"
    syftbox.mkdir()
    do_config = SyftJobConfig(syftbox_folder=syftbox, current_user_email=DO_EMAIL)
    do_client = JobClient(config=do_config)

    do_client.setup_ds_job_folder_as_do(DS_EMAIL)
    job_dir = do_config.get_job_submission_dir(DO_EMAIL, DS_EMAIL, "some.job")
    job_dir.mkdir(parents=True)

    # The grants on inbox/<ds>/ and review/<ds>/ cover the v<n>/<job> subfolder.
    segment = protocol_dir_name(JOB_PROTOCOL_VERSION)
    ctx = SyftPermContext(datasite=syftbox / DO_EMAIL)
    assert ctx.open(
        f"app_data/job/inbox/{DS_EMAIL}/{segment}/some.job/"
    ).has_write_access(DS_EMAIL)
    assert ctx.open(
        f"app_data/job/review/{DS_EMAIL}/{segment}/some.job/state.yaml"
    ).has_read_access(DS_EMAIL)


def test_negotiated_protocol_version_for_peer_raises_on_unknown(tmp_path: Path):
    syftbox = tmp_path / "SyftBox"
    syftbox.mkdir()
    ds_config = SyftJobConfig(syftbox_folder=syftbox, current_user_email=DS_EMAIL)
    current_schema = job_registry.compute_protocol_schema()
    manager = JobClient(
        config=ds_config, peer_schema={DO_EMAIL: current_schema}.get
    ).manager

    # A known peer negotiates the version both sides read.
    assert manager.negotiated_protocol_version_for_peer(DO_EMAIL) == (
        JOB_PROTOCOL_VERSION
    )
    with pytest.raises(MigrationError):
        manager.negotiated_protocol_version_for_peer("stranger@test.org")
    # Opting out assumes the current protocol.
    assert (
        manager.negotiated_protocol_version_for_peer(
            "stranger@test.org", raise_on_unknown=False
        )
        == JOB_PROTOCOL_VERSION
    )
