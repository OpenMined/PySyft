from syft_rds import SyftRDSClientConfig
from syft.sync.connections.drive.mock_drive_service import (
    MockDriveBackingStore,
    MockDriveService,
)
from syft.sync.connections.drive.drive_cached_client import DriveCachedClient
from syft.sync.syftbox_manager import SyftboxManager
from syft.sync.transport.protocols import Transport


def create_configs(
    enclave_email,
    do1_email,
    do2_email,
    ds_email,
    use_in_memory_cache,
    policies,
    enclave_persist_owner_state=False,
) -> tuple:
    """Configs of the enclave, DO1, DO2 and DS, with ``policies`` in that order."""
    enclave_policy, do1_policy, do2_policy, ds_policy = policies
    enclave_config = SyftRDSClientConfig._base_config_for_testing(
        email=enclave_email,
        has_do_role=True,
        has_ds_role=True,
        use_in_memory_cache=use_in_memory_cache,
        persist_owner_state=enclave_persist_owner_state,
        encryption_policy=enclave_policy,
    )
    do1_config = SyftRDSClientConfig._base_config_for_testing(
        email=do1_email,
        has_do_role=True,
        has_ds_role=True,
        use_in_memory_cache=use_in_memory_cache,
        encryption_policy=do1_policy,
    )
    do2_config = SyftRDSClientConfig._base_config_for_testing(
        email=do2_email,
        has_do_role=True,
        has_ds_role=True,
        use_in_memory_cache=use_in_memory_cache,
        encryption_policy=do2_policy,
    )
    ds_config = SyftRDSClientConfig._base_config_for_testing(
        email=ds_email,
        has_ds_role=True,
        use_in_memory_cache=use_in_memory_cache,
        encryption_policy=ds_policy,
    )
    return enclave_config, do1_config, do2_config, ds_config


def create_clients(configs) -> tuple:
    """One client per config, all on one mock Drive."""
    from syft_enclaves.client import SyftEnclaveClient

    backing_store = MockDriveBackingStore()
    return tuple(
        SyftEnclaveClient.from_config(
            c, transport=_mock_transport(backing_store, c.sync.email)
        )
        for c in configs
    )


def _mock_transport(backing_store: MockDriveBackingStore, email: str) -> Transport:
    service = MockDriveService(backing_store, email)
    return SyftboxManager._drive_transport(
        DriveCachedClient.from_service(email, service)
    )


def setup_callbacks(managers: tuple):
    enclave, do1, do2, ds = managers

    # DS file writer callbacks (for all managers with DS role)
    for ds_manager in (enclave, do1, do2, ds):
        ds_manager.file_writer.add_callback(
            "write_file",
            ds_manager.datasite_watcher_syncer.on_file_change,
        )

    # DO event cache callbacks for job handling
    for do_manager in (enclave, do1, do2):
        do_manager.datasite_owner_syncer.event_cache.add_callback(
            "on_event_local_write",
            do_manager.job_file_change_handler._handle_file_change,
        )


def write_versions(managers: tuple):
    for manager in managers:
        manager.peer_manager.write_own_peer_info()


def wire_peers(managers: tuple):
    enclave, do1, do2, ds = managers

    # Each side adds the peers they want to communicate with.
    # load_peers() (called during sync) will auto-upgrade mutual
    # REQUESTED_BY_ME connections to ACCEPTED.
    ds.add_peer(do1.email)
    ds.add_peer(do2.email)
    ds.add_peer(enclave.email)

    enclave.add_peer(do1.email)
    enclave.add_peer(do2.email)
    enclave.add_peer(ds.email)

    for dual_manager in (do1, do2):
        dual_manager.add_peer(enclave.email)
        dual_manager.add_peer(ds.email)

    # load_peers upgrades mutual REQUESTED_BY_ME to ACCEPTED
    for manager in managers:
        manager.load_peers()
