# stdlib
from copy import deepcopy

# third party
from pytest import MonkeyPatch
from result import Err
from result import Ok

# syft absolute
from syft.core.node.new.context import AuthedServiceContext
from syft.core.node.new.metadata_service import MetadataService
from syft.core.node.new.metadata_stash import MetadataStash
from syft.core.node.new.node_metadata import NodeMetadata
from syft.core.node.new.node_metadata import NodeMetadataUpdate
from syft.core.node.new.response import SyftError


def test_metadataservice_get_success(
    monkeypatch: MonkeyPatch,
    metadata_service: MetadataService,
    metadata: NodeMetadata,
    authed_context: AuthedServiceContext,
) -> None:
    mock_stash_get_all_output = [metadata, metadata]
    expected_output = Ok(mock_stash_get_all_output[0])

    def mock_stash_get_all() -> Ok:
        return Ok(mock_stash_get_all_output)

    monkeypatch.setattr(metadata_service.stash, "get_all", mock_stash_get_all)

    response = metadata_service.get(context=authed_context)

    assert isinstance(response.ok(), NodeMetadata)
    assert response == expected_output


def test_metadataservice_get_stash_fail(
    monkeypatch: MonkeyPatch,
    metadata_service: MetadataService,
    metadata: NodeMetadata,
    authed_context: AuthedServiceContext,
) -> None:
    # case 1: we got an empty list from the stash
    response = metadata_service.get(context=authed_context)
    assert isinstance(response, SyftError)
    assert response.message == "No metadata found"

    # case 2: the stash.get_all() function fails
    mock_error_message = "database failure"

    def mock_stash_get_all_error() -> Err:
        return Err(mock_error_message)

    monkeypatch.setattr(metadata_service.stash, "get_all", mock_stash_get_all_error)

    response = metadata_service.get(context=authed_context)
    assert isinstance(response, SyftError)
    assert response.message == mock_error_message


def test_metadataservice_set_success(
    metadata_service: MetadataService,
    metadata: NodeMetadata,
    authed_context: AuthedServiceContext,
) -> None:
    response = metadata_service.set(authed_context, metadata)

    assert response.is_ok() is True
    assert isinstance(response.ok(), NodeMetadata)
    assert response.ok() == metadata


def test_metadataservice_set_fail(
    monkeypatch: MonkeyPatch,
    metadata_service: MetadataService,
    metadata: NodeMetadata,
    authed_context: AuthedServiceContext,
) -> None:
    mock_error_message = "database failure"

    def mock_stash_set_error(a) -> Err:
        return Err(mock_error_message)

    monkeypatch.setattr(metadata_service.stash, "set", mock_stash_set_error)

    response = metadata_service.set(authed_context, metadata)

    assert isinstance(response, SyftError)
    assert response.message == mock_error_message


def add_mock_metadata(
    metadata_stash: MetadataStash, metadata: NodeMetadata
) -> NodeMetadata:
    # create a mock metadata in the stash so that we can update it
    result = metadata_stash.partition.set(metadata)
    assert result.is_ok()

    created_metadata = result.ok()
    assert created_metadata is not None

    return created_metadata


def test_metadataservice_update_success(
    monkeypatch: MonkeyPatch,
    metadata_stash: MetadataStash,
    metadata_service: MetadataService,
    metadata: NodeMetadata,
    update_metadata: NodeMetadataUpdate,
    authed_context: AuthedServiceContext,
) -> None:
    # add a mock metadata to the stash
    mock_metadata = add_mock_metadata(metadata_stash, metadata)

    # get a new metadata according to update_metadata
    new_metadata = deepcopy(metadata)
    update_kwargs = update_metadata.to_dict(exclude_none=True).items()
    for field_name, value in update_kwargs:
        setattr(new_metadata, field_name, value)

    assert new_metadata != metadata
    assert new_metadata != mock_metadata
    assert mock_metadata == metadata

    mock_stash_get_all_output = [mock_metadata, mock_metadata]

    def mock_stash_get_all() -> Ok:
        return Ok(mock_stash_get_all_output)

    monkeypatch.setattr(metadata_service.stash, "get_all", mock_stash_get_all)

    # update the metadata in the metadata stash using metadata_service
    response = metadata_service.update(authed_context, update_metadata)
    updated_metadata = response.ok()[0]
    not_updated_metadata = response.ok()[1]

    assert response.is_ok() is True
    assert len(response.ok()) == len(mock_stash_get_all_output)
    assert updated_metadata == new_metadata  # the first metadata is updated
    assert not_updated_metadata == metadata  # the second metadata is not updated


def test_metadataservice_update_stash_get_all_fail(
    monkeypatch: MonkeyPatch,
    metadata_service: MetadataService,
    update_metadata: NodeMetadataUpdate,
    authed_context: AuthedServiceContext,
) -> None:
    # the stash.get_all() function fails
    mock_error_message = "database failure"

    def mock_stash_get_all_error() -> Err:
        return Err(mock_error_message)

    monkeypatch.setattr(metadata_service.stash, "get_all", mock_stash_get_all_error)
    response = metadata_service.update(authed_context, update_metadata)

    assert isinstance(response, SyftError)
    assert response.message == mock_error_message


def test_metadataservice_update_stash_empty(
    metadata_service: MetadataService,
    update_metadata: NodeMetadataUpdate,
    authed_context: AuthedServiceContext,
) -> None:
    response = metadata_service.update(authed_context, update_metadata)

    assert isinstance(response, SyftError)
    assert response.message == "No metadata found"


def test_metadataservice_update_fail(
    monkeypatch: MonkeyPatch,
    metadata: NodeMetadata,
    metadata_service: MetadataService,
    update_metadata: NodeMetadataUpdate,
    authed_context: AuthedServiceContext,
) -> None:
    # the stash has a metadata but we could not update it (the stash.update() function fails)

    mock_stash_get_all_output = [metadata, metadata]

    def mock_stash_get_all() -> Ok:
        return Ok(mock_stash_get_all_output)

    monkeypatch.setattr(metadata_service.stash, "get_all", mock_stash_get_all)

    mock_update_error_message = "Failed to update obj NodeMetadata"

    def mock_stash_update_error(update_metadata: NodeMetadata) -> Err:
        return Err(mock_update_error_message)

    monkeypatch.setattr(metadata_service.stash, "update", mock_stash_update_error)

    response = metadata_service.update(authed_context, update_metadata)

    assert isinstance(response, SyftError)
    assert response.message == mock_update_error_message
