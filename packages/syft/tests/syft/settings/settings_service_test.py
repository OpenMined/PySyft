# stdlib
from copy import deepcopy

# third party
from pytest import MonkeyPatch
from result import Err
from result import Ok

# syft absolute
from syft.service.settings.settings import NodeSettings
from syft.service.settings.settings import NodeSettingsUpdate
from syft.service.settings.settings_service import SettingsService
from syft.service.settings.settings_stash import SettingsStash
from syft.types.context import AuthedServiceContext
from syft.types.credentials import SyftVerifyKey
from syft.types.response import SyftError


def test_settingsservice_get_success(
    monkeypatch: MonkeyPatch,
    settings_service: SettingsService,
    settings: NodeSettings,
    authed_context: AuthedServiceContext,
) -> None:
    mock_stash_get_all_output = [settings, settings]
    expected_output = Ok(mock_stash_get_all_output[0])

    def mock_stash_get_all(credentials) -> Ok:
        return Ok(mock_stash_get_all_output)

    monkeypatch.setattr(settings_service.stash, "get_all", mock_stash_get_all)

    response = settings_service.get(context=authed_context)

    assert isinstance(response.ok(), NodeSettings)
    assert response == expected_output


def test_settingsservice_get_stash_fail(
    monkeypatch: MonkeyPatch,
    settings_service: SettingsService,
    authed_context: AuthedServiceContext,
) -> None:
    def mock_empty_stash(credentials):
        return Ok([])

    monkeypatch.setattr(settings_service.stash, "get_all", mock_empty_stash)

    # case 1: we got an empty list from the stash
    response = settings_service.get(context=authed_context)
    assert isinstance(response, SyftError)
    assert response.message == "No settings found"

    # case 2: the stash.get_all() function fails
    mock_error_message = "database failure"

    def mock_stash_get_all_error(credentials) -> Err:
        return Err(mock_error_message)

    monkeypatch.setattr(settings_service.stash, "get_all", mock_stash_get_all_error)

    response = settings_service.get(context=authed_context)
    assert isinstance(response, SyftError)
    assert response.message == mock_error_message


def test_settingsservice_set_success(
    settings_service: SettingsService,
    settings: NodeSettings,
    authed_context: AuthedServiceContext,
) -> None:
    response = settings_service.set(authed_context, settings)

    assert response.is_ok() is True
    assert isinstance(response.ok(), NodeSettings)
    assert response.ok() == settings


def test_settingsservice_set_fail(
    monkeypatch: MonkeyPatch,
    settings_service: SettingsService,
    settings: NodeSettings,
    authed_context: AuthedServiceContext,
) -> None:
    mock_error_message = "database failure"

    def mock_stash_set_error(credentials, a) -> Err:
        return Err(mock_error_message)

    monkeypatch.setattr(settings_service.stash, "set", mock_stash_set_error)

    response = settings_service.set(authed_context, settings)

    assert isinstance(response, SyftError)
    assert response.message == mock_error_message


def add_mock_settings(
    root_verify_key: SyftVerifyKey,
    settings_stash: SettingsStash,
    settings: NodeSettings,
) -> NodeSettings:
    # create a mock settings in the stash so that we can update it
    result = settings_stash.partition.set(root_verify_key, settings)
    assert result.is_ok()

    created_settings = result.ok()
    assert created_settings is not None

    return created_settings


def test_settingsservice_update_success(
    root_verify_key,
    monkeypatch: MonkeyPatch,
    settings_stash: SettingsStash,
    settings_service: SettingsService,
    settings: NodeSettings,
    update_settings: NodeSettingsUpdate,
    authed_context: AuthedServiceContext,
) -> None:
    # add a mock settings to the stash
    mock_settings = add_mock_settings(
        authed_context.credentials, settings_stash, settings
    )

    # get a new settings according to update_settings
    new_settings = deepcopy(settings)
    update_kwargs = update_settings.to_dict(exclude_none=True).items()
    for field_name, value in update_kwargs:
        setattr(new_settings, field_name, value)

    assert new_settings != settings
    assert new_settings != mock_settings
    assert mock_settings == settings

    mock_stash_get_all_output = [mock_settings, mock_settings]

    def mock_stash_get_all(root_verify_key) -> Ok:
        return Ok(mock_stash_get_all_output)

    monkeypatch.setattr(settings_service.stash, "get_all", mock_stash_get_all)

    # update the settings in the settings stash using settings_service
    response = settings_service.update(authed_context, update_settings)
    print(response)
    updated_settings = response.ok()[0]
    not_updated_settings = response.ok()[1]

    assert response.is_ok() is True
    assert len(response.ok()) == len(mock_stash_get_all_output)
    assert updated_settings == new_settings  # the first settings is updated
    assert not_updated_settings == settings  # the second settings is not updated


def test_settingsservice_update_stash_get_all_fail(
    monkeypatch: MonkeyPatch,
    settings_service: SettingsService,
    update_settings: NodeSettingsUpdate,
    authed_context: AuthedServiceContext,
) -> None:
    # the stash.get_all() function fails
    mock_error_message = "database failure"

    def mock_stash_get_all_error(credentials) -> Err:
        return Err(mock_error_message)

    monkeypatch.setattr(settings_service.stash, "get_all", mock_stash_get_all_error)
    response = settings_service.update(authed_context, update_settings)

    assert isinstance(response, SyftError)
    assert response.message == mock_error_message


def test_settingsservice_update_stash_empty(
    settings_service: SettingsService,
    update_settings: NodeSettingsUpdate,
    authed_context: AuthedServiceContext,
) -> None:
    response = settings_service.update(authed_context, update_settings)

    assert isinstance(response, SyftError)
    assert response.message == "No settings found"


def test_settingsservice_update_fail(
    monkeypatch: MonkeyPatch,
    settings: NodeSettings,
    settings_service: SettingsService,
    update_settings: NodeSettingsUpdate,
    authed_context: AuthedServiceContext,
) -> None:
    # the stash has a settings but we could not update it (the stash.update() function fails)

    mock_stash_get_all_output = [settings, settings]

    def mock_stash_get_all(credentials) -> Ok:
        return Ok(mock_stash_get_all_output)

    monkeypatch.setattr(settings_service.stash, "get_all", mock_stash_get_all)

    mock_update_error_message = "Failed to update obj NodeMetadata"

    def mock_stash_update_error(credentials, update_settings: NodeSettings) -> Err:
        return Err(mock_update_error_message)

    monkeypatch.setattr(settings_service.stash, "update", mock_stash_update_error)

    response = settings_service.update(authed_context, update_settings)

    assert isinstance(response, SyftError)
    assert response.message == mock_update_error_message
