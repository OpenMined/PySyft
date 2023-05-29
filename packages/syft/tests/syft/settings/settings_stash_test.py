# third party

# syft absolute
from syft.service.settings.settings import NodeSettings
from syft.service.settings.settings import NodeSettingsUpdate
from syft.service.settings.settings_stash import SettingsStash


def add_mock_settings(
    root_verify_key, settings_stash: SettingsStash, settings: NodeSettings
) -> NodeSettings:
    # prepare: add mock settings
    result = settings_stash.partition.set(root_verify_key, settings)
    assert result.is_ok()

    created_settings = result.ok()
    assert created_settings is not None

    return created_settings


def test_settingsstash_set(
    root_verify_key, settings_stash: SettingsStash, settings: NodeSettings
) -> None:
    result = settings_stash.set(root_verify_key, settings)
    assert result.is_ok()

    created_settings = result.ok()
    assert isinstance(created_settings, NodeSettings)
    assert created_settings == settings
    assert settings.id in settings_stash.partition.data


def test_settingsstash_update(
    root_verify_key,
    settings_stash: SettingsStash,
    settings: NodeSettings,
    update_settings: NodeSettingsUpdate,
) -> None:
    # prepare: add a mock settings
    mock_settings = add_mock_settings(root_verify_key, settings_stash, settings)

    # update mock_settings according to update_settings
    update_kwargs = update_settings.to_dict(exclude_none=True).items()
    for field_name, value in update_kwargs:
        setattr(mock_settings, field_name, value)

    # update the settings in the stash
    result = settings_stash.update(root_verify_key, settings=mock_settings)

    assert result.is_ok()
    updated_settings = result.ok()
    assert isinstance(updated_settings, NodeSettings)
    assert mock_settings == updated_settings
