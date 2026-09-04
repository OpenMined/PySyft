"""A job written for a protocol-0 peer reaches that peer and reads back.

The other tests in this folder stop at the negotiated version. They assert which
protocol the two sides agree on, not that a job written at that protocol arrives
and reads. That seam is where the dataset transport broke: negotiation chose a
layout the delivery path could not carry.

This test drives the whole path: the peer advertises job protocol 0, the sender
negotiates down, writes the flat layout, syncs, and the receiver finds and reads
the job through its own scan.
"""

from pathlib import Path

import pytest
from syft_rds import SyftRDSClient
from syft_migration import ProtocolSchema

from tests.unit.utils import create_test_project_folder


def _job_schema(protocol_version: str) -> ProtocolSchema:
    # The slim form a peer advertises in its VersionInfo.
    return ProtocolSchema(
        protocol_name="syft-job",
        version=protocol_version,
        supported_versions={"JobState": ["1"], "JobSubmissionMetadata": ["1"]},
    )


@pytest.fixture
def pair():
    return SyftRDSClient.pair_with_mock_drive_service_connection(
        use_in_memory_cache=False,
        sync_automatically=False,
    )


def _submit(ds_manager, do_manager, job_name: str) -> Path:
    project_dir = create_test_project_folder(with_pyproject=False)
    ds_manager.submit_python_job(
        user=do_manager.email,
        code_path=str(project_dir),
        job_name=job_name,
        entrypoint="main.py",
    )
    do_manager.sync()
    return project_dir


def test_a_job_for_a_protocol0_peer_uses_the_flat_layout(pair):
    ds_manager, do_manager = pair
    # The DO advertises job protocol 0, as a client of 0.1.38 or earlier does.
    ds_manager.peer_manager.live_peer_schemas("syft-job")[do_manager.email] = (
        _job_schema("0")
    )

    ref = ds_manager.job_client.manager.new_submission_ref(do_manager.email, "skew.job")
    assert ref.protocol_version == "0"
    assert "/v0/" not in str(ref) and "/v1/" not in str(ref), (
        "protocol 0 is the flat layout, so the path carries no v<n> segment"
    )


def test_a_job_for_a_protocol0_peer_arrives_and_reads(pair):
    ds_manager, do_manager = pair
    ds_manager.peer_manager.live_peer_schemas("syft-job")[do_manager.email] = (
        _job_schema("0")
    )

    _submit(ds_manager, do_manager, "skew.job")

    # The receiver scans every layout it knows, so it finds the flat one.
    assert [job.name for job in do_manager.jobs] == ["skew.job"]
    found = do_manager.job_client.manager.find_submission_ref(
        do_manager.email, "skew.job"
    )
    assert found.protocol_version == "0"


def test_a_job_for_a_current_peer_still_uses_the_versioned_layout(pair):
    # The control: without a protocol-0 peer the sender keeps the current layout,
    # so the test above measures negotiation and not a broken default.
    ds_manager, do_manager = pair
    _submit(ds_manager, do_manager, "current.job")

    found = do_manager.job_client.manager.find_submission_ref(
        do_manager.email, "current.job"
    )
    assert found.protocol_version != "0"
    assert [job.name for job in do_manager.jobs] == ["current.job"]
