"""Tests for the self-contained SyftRDSClient product."""

from syft_rds import SyftRDSClient


def test_rds_layer_supplies_collection_specs():
    """The composed DS sync engine's watcher cache received the dataset spec
    from the RDS layer (the RDS -> generic-engine spec-injection seam)."""
    from syft_datasets.dataset_manager import DATASET_COLLECTION_PREFIX

    ds, do = SyftRDSClient.pair_with_mock_drive_service_connection(
        use_in_memory_cache=False
    )

    watcher_cache = ds.sync_engine.datasite_watcher_syncer.datasite_watcher_cache
    specs = watcher_cache.collection_specs

    assert len(specs) > 0
    assert any(spec.prefix == DATASET_COLLECTION_PREFIX for spec in specs)
