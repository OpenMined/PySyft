from syft_client.sync.syftbox_manager import SyftboxManagerConfig
from syft_client.sync.connections.drive.mock_drive_service import (
    MockDriveBackingStore,
    MockDriveService,
)
from syft_client.sync.connections.drive.gdrive_transport import GDriveConnection


def create_configs(
    enclave_email, do1_email, do2_email, ds_email, use_in_memory_cache
) -> tuple:
    enclave_config = SyftboxManagerConfig._base_config_for_testing(
        email=enclave_email,
        has_do_role=True,
        has_ds_role=True,
        use_in_memory_cache=use_in_memory_cache,
    )
    do1_config = SyftboxManagerConfig._base_config_for_testing(
        email=do1_email,
        has_do_role=True,
        has_ds_role=True,
        use_in_memory_cache=use_in_memory_cache,
    )
    do2_config = SyftboxManagerConfig._base_config_for_testing(
        email=do2_email,
        has_do_role=True,
        has_ds_role=True,
        use_in_memory_cache=use_in_memory_cache,
    )
    ds_config = SyftboxManagerConfig._base_config_for_testing(
        email=ds_email,
        has_ds_role=True,
        use_in_memory_cache=use_in_memory_cache,
    )
    return enclave_config, do1_config, do2_config, ds_config


def create_clients(configs) -> tuple:
    from syft_enclaves.client import SyftEnclaveClient

    return tuple(SyftEnclaveClient.from_config(c) for c in configs)


def setup_connections(managers: tuple):
    backing_store = MockDriveBackingStore()

    for manager in managers:
        service = MockDriveService(backing_store, manager.email)
        connection = GDriveConnection.from_service(manager.email, service)
        manager._add_connection(connection)


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
        manager.peer_manager.write_own_version()


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
