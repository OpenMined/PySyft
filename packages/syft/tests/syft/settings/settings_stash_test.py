# syft absolute
from syft.service.settings.settings import ServerSettings
from syft.service.settings.settings import ServerSettingsUpdate
from syft.service.settings.settings_stash import SettingsStash


def test_settingsstash_set(
    root_verify_key,
    settings_stash: SettingsStash,
    settings: ServerSettings,
    update_settings: ServerSettingsUpdate,
) -> None:
    created_settings = settings_stash.set(root_verify_key, settings).unwrap()
    assert isinstance(created_settings, ServerSettings)
    assert created_settings == settings
    assert settings_stash.exists(root_verify_key, settings.id)

    # update mock_settings according to update_settings
    update_kwargs = update_settings.to_dict(exclude_empty=True).items()
    for field_name, value in update_kwargs:
        setattr(settings, field_name, value)

    # update the settings in the stash
    updated_settings = settings_stash.update(root_verify_key, obj=settings).unwrap()
    assert isinstance(updated_settings, ServerSettings)
    assert settings == updated_settings
