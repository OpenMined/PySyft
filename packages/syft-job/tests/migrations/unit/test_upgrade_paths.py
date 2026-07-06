"""Every registered object version can be upgraded to the latest version."""

from syft_job.migrations import job_registry


def test_every_version_has_upgrade_path_to_latest():
    assert job_registry.objects  # sanity: the registry is populated

    for canonical_name, versions in job_registry.objects.items():
        for version in versions:
            assert job_registry.has_upgradeable_path_to_latest(
                canonical_name=canonical_name, from_version=version
            ), f"No upgrade path for {canonical_name!r} v{version} to latest"
