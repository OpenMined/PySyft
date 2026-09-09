"""Every registered object version can migrate up to latest and down to any lower."""

from syft_datasets.migrations import dataset_registry


def test_every_version_has_upgrade_path_to_latest():
    assert dataset_registry.objects  # sanity: the registry is populated

    for canonical_name, versions in dataset_registry.objects.items():
        for version in versions:
            assert dataset_registry.has_upgradeable_path_to_latest(
                canonical_name=canonical_name, from_version=version
            ), f"No upgrade path for {canonical_name!r} v{version} to latest"


def test_every_version_has_downgrade_path_to_all_lower_versions():
    assert dataset_registry.objects  # sanity: the registry is populated

    for canonical_name, versions in dataset_registry.objects.items():
        for higher in versions:
            for lower in versions:
                if lower >= higher:
                    continue
                assert dataset_registry.has_migration_path(
                    canonical_name=canonical_name,
                    from_version=higher,
                    to_version=lower,
                ), f"No downgrade path for {canonical_name!r} v{higher} to v{lower}"
