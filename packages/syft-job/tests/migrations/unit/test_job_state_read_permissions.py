"""JobStateV2 carries the read permissions; a V1 reader never sees them."""

from syft_job.migrations import job_registry
from syft_job.models import JobStateV1, JobStateV2, JobStatus
from syft_migration import MigrationService

READ = {"stdout.txt": ["ds@test.org"]}


def test_v1_state_upgrades_with_no_read_permissions():
    v1 = JobStateV1(status=JobStatus.DONE)
    upgraded = MigrationService(registry=job_registry).migrate(v1, "2")

    assert isinstance(upgraded, JobStateV2)
    assert upgraded.status == JobStatus.DONE
    assert upgraded.requested_read_permissions == {}
    assert upgraded.approved_read_permissions == {}


def test_v2_state_downgrades_without_read_permissions():
    v2 = JobStateV2(
        status=JobStatus.DONE,
        requested_read_permissions=READ,
        approved_read_permissions=READ,
    )
    downgraded = MigrationService(registry=job_registry).migrate(v2, "1")

    assert type(downgraded) is JobStateV1
    assert downgraded.status == JobStatus.DONE
    assert "approved_read_permissions" not in downgraded.disk_dict()
