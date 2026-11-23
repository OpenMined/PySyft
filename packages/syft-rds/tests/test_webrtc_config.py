"""The syft-rds config of a WebRTC login."""

from syft_rds.config import SyftRDSClientConfig

from syft.sync.connections.webrtc.webrtc_config import WebRTCConfig


def test_webrtc_config_registers_no_dataset_specs():
    # A dataset spec would make a DS sync delete its local dataset copies.
    webrtc = WebRTCConfig(firebase_project_id="syft-test", firebase_api_key="k")

    config = SyftRDSClientConfig.for_jupyter(
        "ds@example.com", has_ds_role=True, webrtc=webrtc
    )

    sync = config.sync
    watcher_config = sync.datasite_watcher_syncer_config.datasite_watcher_cache_config
    assert watcher_config.collection_specs == []
    assert sync.datasite_owner_syncer_config.collection_specs == []
