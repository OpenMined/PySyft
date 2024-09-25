# stdlib
from contextlib import contextmanager
import secrets

# third party
import faker
import numpy as np
import pytest
import yaml

# syft absolute
import syft as sy
from syft.client.datasite_client import DatasiteClient
from syft.service.migration.object_migration_state import MigrationData
from syft.service.response import SyftSuccess
from syft.service.user.user import User
from syft.types.errors import SyftException


def register_ds(client):
    f = faker.Faker()

    email = f.email()
    password = secrets.token_urlsafe(16)
    client.register(
        name=f.name(),
        email=email,
        password=password,
        password_verify=password,
    )
    return client.login(email=email, password=password)


def create_dataset(client):
    mock = np.random.random(5)
    private = np.random.random(5)

    dataset = sy.Dataset(
        name=sy.util.util.random_name().lower(),
        description="Lorem ipsum dolor sit amet, consectetur adipiscing elit",
        asset_list=[
            sy.Asset(
                name="numpy-data",
                mock=mock,
                data=private,
                shape=private.shape,
                mock_is_real=True,
            )
        ],
    )

    client.upload_dataset(dataset)
    return dataset


def make_request(client: DatasiteClient) -> DatasiteClient:
    @sy.syft_function_single_use()
    def compute() -> int:
        return 42

    _ = client.code.request_code_execution(compute)


def prepare_data(client: DatasiteClient) -> None:
    # Create DS, upload dataset, create + approve + execute single request
    ds_client = register_ds(client)
    create_dataset(client)

    @sy.syft_function_single_use()
    def compute() -> int:
        return 42

    ds_client.code.request_code_execution(compute)

    client.requests[0].approve()

    result = ds_client.code.compute()
    assert result.get() == 42


def test_get_migration_data(worker, tmp_path):
    # Tests a full data dump for migration
    # TODO current prepare_data is a small scenario, add more data + edge-cases
    client = worker.root_client
    prepare_data(client)

    migration_data = client.get_migration_data()
    assert isinstance(migration_data, MigrationData)

    # Admin + data scientist
    assert len(migration_data.store_objects[User]) == 2

    # Check if all blobs are there
    blob_ids = {blob.id for blob in migration_data.blob_storage_objects}
    assert blob_ids == set(migration_data.blobs.keys())

    # Save + load
    blob_path = tmp_path / "migration.blob"
    yaml_path = tmp_path / "migration.yaml"
    migration_data.save(blob_path, yaml_path)

    loaded_migration_data = MigrationData.from_file(blob_path)

    with open(yaml_path) as f:
        loaded_migration_yaml = yaml.safe_load(f)

    assert isinstance(loaded_migration_data, MigrationData)
    assert loaded_migration_data.num_objects == migration_data.num_objects
    assert loaded_migration_data.num_action_objects == migration_data.num_action_objects
    assert loaded_migration_data.blobs.keys() == migration_data.blobs.keys()

    assert loaded_migration_yaml == migration_data.make_migration_config()


@contextmanager
def named_worker_context(name):
    # required to launch worker with same name twice within the same test + ensure cleanup
    worker = sy.Worker.named(name=name, db_url="sqlite://")
    try:
        yield worker
    finally:
        worker.cleanup()


def test_data_migration_same_version(tmp_path):
    server_name = secrets.token_hex(8)
    blob_path = tmp_path / "migration.blob"
    yaml_path = tmp_path / "migration.yaml"

    # Setup + save migration data
    with named_worker_context(server_name) as first_worker:
        prepare_data(first_worker.root_client)
        first_migration_data = first_worker.root_client.get_migration_data()
        first_migration_data.save(blob_path, yaml_path)

    # Load migration data on wrong worker
    with named_worker_context(secrets.token_hex(8)) as wrong_worker:
        with pytest.raises(SyftException):
            result = wrong_worker.root_client.load_migration_data(blob_path)

    # Load migration data on correct worker
    # NOTE worker is correct because admin keys and server id are derived from server name,
    # so they match the first worker
    with named_worker_context(server_name) as migration_worker:
        client = migration_worker.root_client

        # DB is new, no DS registered yet
        assert len(client.users.get_all()) == 1

        assert migration_worker.id == first_migration_data.server_uid
        assert migration_worker.verify_key == first_migration_data.root_verify_key

        result = migration_worker.root_client.load_migration_data(blob_path)
        assert isinstance(result, SyftSuccess)

        assert len(client.users.get_all()) == 2
        assert len(client.requests.get_all()) == 1
        assert len(client.datasets.get_all()) == 1
        assert len(client.code.get_all()) == 1
