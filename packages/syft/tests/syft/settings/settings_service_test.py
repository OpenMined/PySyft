# stdlib
from copy import deepcopy
from datetime import datetime
from typing import NoReturn
from unittest import mock
from uuid import uuid4

# third party
from faker import Faker
import pytest
from pytest import MonkeyPatch

# syft absolute
import syft
from syft.abstract_server import ServerSideType
from syft.client.datasite_client import DatasiteClient
from syft.server.credentials import SyftSigningKey
from syft.server.credentials import SyftVerifyKey
from syft.service.context import AuthedServiceContext
from syft.service.notifier.notifier import NotifierSettings
from syft.service.notifier.notifier_stash import NotifierStash
from syft.service.response import SyftSuccess
from syft.service.service import _SIGNATURE_ERROR_MESSAGE
from syft.service.settings.settings import ServerSettings
from syft.service.settings.settings import ServerSettingsUpdate
from syft.service.settings.settings_service import (
    _NOTIFICATIONS_ENABLED_WIHOUT_CREDENTIALS_ERROR,
)
from syft.service.settings.settings_service import SettingsService
from syft.service.settings.settings_stash import SettingsStash
from syft.service.user.user import UserPrivateKey
from syft.service.user.user import UserView
from syft.service.user.user_roles import ServiceRole
from syft.store.document_store_errors import NotFoundException
from syft.store.document_store_errors import StashException
from syft.types.errors import SyftException
from syft.types.result import as_result


def test_settingsservice_get_success(
    monkeypatch: MonkeyPatch,
    settings_service: SettingsService,
    settings: ServerSettings,
    authed_context: AuthedServiceContext,
) -> None:
    mock_stash_get_all_output = [settings, settings]
    expected_output = mock_stash_get_all_output[0]

    @as_result(SyftException)
    def mock_stash_get_all(credentials) -> list[ServerSettings]:
        return mock_stash_get_all_output

    monkeypatch.setattr(settings_service.stash, "get_all", mock_stash_get_all)

    response = settings_service.get(context=authed_context)

    assert isinstance(response, ServerSettings)
    assert response == expected_output


def test_settingsservice_get_stash_fail(
    monkeypatch: MonkeyPatch,
    settings_service: SettingsService,
    authed_context: AuthedServiceContext,
) -> None:
    @as_result(StashException)
    def mock_empty_stash(credentials) -> list[ServerSettings]:
        return []

    monkeypatch.setattr(settings_service.stash, "get_all", mock_empty_stash)

    # case 1: we got an empty list from the stash
    with pytest.raises(NotFoundException) as exc:
        settings_service.get(context=authed_context)

    assert exc.type == NotFoundException
    assert exc.value.public_message == "No settings found"

    # case 2: the stash.get_all() function fails
    mock_error_message = "database failure"

    @as_result(StashException)
    def mock_stash_get_all_error(credentials) -> NoReturn:
        raise StashException(public_message=mock_error_message)

    monkeypatch.setattr(settings_service.stash, "get_all", mock_stash_get_all_error)

    with pytest.raises(StashException) as exc:
        settings_service.get(context=authed_context)

    assert exc.type == StashException
    assert exc.value.public_message == mock_error_message


def test_settingsservice_set_success(
    settings_service: SettingsService,
    settings: ServerSettings,
    authed_context: AuthedServiceContext,
) -> None:
    response = settings_service.set(authed_context, settings)
    assert isinstance(response, ServerSettings)
    response.syft_client_verify_key = None
    response.syft_server_location = None
    response.pwd_token_config.syft_client_verify_key = None
    response.pwd_token_config.syft_server_location = None
    response.welcome_markdown.syft_client_verify_key = None
    response.welcome_markdown.syft_server_location = None
    assert response == settings


def add_mock_settings(
    root_verify_key: SyftVerifyKey,
    settings_stash: SettingsStash,
    settings: ServerSettings,
) -> ServerSettings:
    # create a mock settings in the stash so that we can update it
    result = settings_stash.set(root_verify_key, settings)
    assert result.is_ok()

    created_settings = result.ok()
    assert created_settings is not None

    return created_settings


def test_settingsservice_update_success(
    root_verify_key,
    monkeypatch: MonkeyPatch,
    settings_stash: SettingsStash,
    settings_service: SettingsService,
    settings: ServerSettings,
    update_settings: ServerSettingsUpdate,
    authed_context: AuthedServiceContext,
    notifier_stash: NotifierStash,
) -> None:
    # add a mock settings to the stash
    mock_settings = settings_stash.set(authed_context.credentials, settings).unwrap()

    # get a new settings according to update_settings
    new_settings = deepcopy(settings)
    update_kwargs = update_settings.to_dict(exclude_empty=True).items()
    for field_name, value in update_kwargs:
        setattr(new_settings, field_name, value)

    assert new_settings != settings
    assert new_settings != mock_settings
    assert mock_settings == settings

    class MockNotifierService:
        def __init__(self, stash):
            self.stash = stash

        def set_notifier_active_to_false(self, context) -> SyftSuccess:
            return SyftSuccess(message="Notifier mocked to True")

        def settings(self, context):
            return NotifierSettings()

    mock_notifier_service = MockNotifierService(stash=notifier_stash)

    def mock_get_service(service_name: str):
        if service_name == "notifierservice":
            return mock_notifier_service
        raise ValueError(f"Unknown service: {service_name}")

    monkeypatch.setattr(authed_context.server, "get_service", mock_get_service)

    # update the settings in the settings stash using settings_service
    response = settings_service.update(context=authed_context, settings=update_settings)

    assert isinstance(response, SyftSuccess)


def test_settingsservice_update_stash_empty(
    settings_service: SettingsService,
    update_settings: ServerSettingsUpdate,
    authed_context: AuthedServiceContext,
) -> None:
    with pytest.raises(NotFoundException) as exc:
        settings_service.update(context=authed_context, settings=update_settings)
        assert exc.value.public_message == "Server settings not found"


def test_settingsservice_update_fail(
    monkeypatch: MonkeyPatch,
    settings: ServerSettings,
    settings_service: SettingsService,
    update_settings: ServerSettingsUpdate,
    authed_context: AuthedServiceContext,
    notifier_stash: NotifierStash,
) -> None:
    # the stash has a settings but we could not update it (the stash.update() function fails)

    mock_stash_get_all_output = [settings, settings]

    @as_result(StashException)
    def mock_stash_get_all(credentials, **kwargs) -> list[ServerSettings]:
        return mock_stash_get_all_output

    monkeypatch.setattr(settings_service.stash, "get_all", mock_stash_get_all)

    mock_update_error_message = "Failed to update obj ServerMetadata"

    @as_result(StashException)
    def mock_stash_update_error(credentials, obj: ServerSettings) -> NoReturn:
        raise StashException(public_message=mock_update_error_message)

    monkeypatch.setattr(settings_service.stash, "update", mock_stash_update_error)

    # Mock the get_service method to return a mocked notifier_service with the notifier_stash
    class MockNotifierService:
        def __init__(self, stash):
            self.stash = stash

        def set_notifier_active_to_false(self, context) -> SyftSuccess:
            return SyftSuccess(message="Notifier mocked to False")

        def settings(self, context):
            return NotifierSettings()

    mock_notifier_service = MockNotifierService(stash=notifier_stash)

    def mock_get_service(service_name: str):
        if service_name == "notifierservice":
            return mock_notifier_service
        raise ValueError(f"Unknown service: {service_name}")

    monkeypatch.setattr(authed_context.server, "get_service", mock_get_service)

    with pytest.raises(StashException) as _:
        settings_service.update(context=authed_context, settings=update_settings)


def test_settings_allow_guest_registration(
    monkeypatch: MonkeyPatch, faker: Faker
) -> None:
    verify_key = SyftSigningKey.generate().verify_key
    mock_server_settings = ServerSettings(
        name=faker.name(),
        verify_key=verify_key,
        highest_version=1,
        lowest_version=2,
        syft_version=syft.__version__,
        signup_enabled=False,
        admin_email="info@openmined.org",
        server_side_type=ServerSideType.LOW_SIDE,
        show_warnings=False,
        deployed_on=datetime.now().date().strftime("%m/%d/%Y"),
        association_request_auto_approval=False,
        notifications_enabled=False,
    )

    with mock.patch(
        "syft.Worker.settings",
        new_callable=mock.PropertyMock,
        return_value=mock_server_settings,
    ):
        worker = syft.Worker.named(name=faker.name(), reset=True, db_url="sqlite://")
        guest_datasite_client = worker.guest_client
        root_datasite_client = worker.root_client

        email1 = faker.email()
        email2 = faker.email()

        response_1 = root_datasite_client.register(
            email=email1, password="joker123", password_verify="joker123", name="Joker"
        )

        assert isinstance(response_1, SyftSuccess)
        assert isinstance(response_1.value, UserPrivateKey)

        # by default, the guest client can't register new user
        with pytest.raises(SyftException) as exc:
            guest_datasite_client.register(
                email=email2,
                password="harley123",
                password_verify="harley123",
                name="Harley",
            )

        expected_err_msg = "You have no permission to create an account. Please contact the Datasite owner."
        assert exc.value.public_message == expected_err_msg
        assert any(user.email == email1 for user in root_datasite_client.users)

    # only after the root client enable other users to signup, they can
    mock_server_settings.signup_enabled = True
    with mock.patch(
        "syft.Worker.settings",
        new_callable=mock.PropertyMock,
        return_value=mock_server_settings,
    ):
        worker = syft.Worker.named(name=faker.name(), reset=True, db_url="sqlite://")
        guest_datasite_client = worker.guest_client
        root_datasite_client = worker.root_client

        password = faker.email()

        response_3 = guest_datasite_client.register(
            email=email2,
            password=password,
            password_verify=password,
            name=faker.name(),
        )

        # FIX: SyftSuccess .value... let's have it in the response instead
        assert isinstance(response_3.value, UserPrivateKey)
        assert any(user.email == email2 for user in root_datasite_client.users)


def test_settings_user_register_for_role(monkeypatch: MonkeyPatch, faker: Faker):
    # Mock patch this env variable to remove race conditions
    # where signup is enabled.

    def get_mock_client(faker, root_client, role):
        email = faker.email()
        password = uuid4().hex

        result = root_client.users.create(
            name=faker.name(),
            email=email,
            role=role,
            password=password,
            password_verify=password,
        )
        assert type(result) == UserView

        guest_client = root_client.guest()
        return guest_client.login(email=email, password=password)

    verify_key = SyftSigningKey.generate().verify_key
    mock_server_settings = ServerSettings(
        name=faker.name(),
        verify_key=verify_key,
        highest_version=1,
        lowest_version=2,
        syft_version=syft.__version__,
        signup_enabled=False,
        admin_email="info@openmined.org",
        server_side_type=ServerSideType.LOW_SIDE,
        show_warnings=False,
        deployed_on=datetime.now().date().strftime("%m/%d/%Y"),
        association_request_auto_approval=False,
        notifications_enabled=False,
    )

    with mock.patch(
        "syft.Worker.settings",
        new_callable=mock.PropertyMock,
        return_value=mock_server_settings,
    ):
        worker = syft.Worker.named(name=faker.name(), reset=True, db_url="sqlite://")
        root_client = worker.root_client

        emails_added = []
        for role in [ServiceRole.DATA_OWNER, ServiceRole.ADMIN]:
            client = get_mock_client(faker=faker, root_client=root_client, role=role)
            email = faker.email()
            result = client.register(
                name=faker.name(),
                email=email,
                password="password",
                password_verify="password",
            )

            assert isinstance(result, SyftSuccess)
            assert isinstance(result.value, UserPrivateKey)
            emails_added.append(email)

        ds_client = get_mock_client(
            faker=faker, root_client=root_client, role=ServiceRole.DATA_SCIENTIST
        )

        with pytest.raises(SyftException) as exc:
            ds_client.register(
                name=faker.name(),
                email=faker.email(),
                password="password",
                password_verify="password",
            )

        error_msg = "You have no permission to create an account. Please contact the Datasite owner."
        assert exc.type is SyftException
        assert exc.value.public_message == error_msg

        users_created_count = sum(
            [u.email in emails_added for u in root_client.users.get_all()]
        )
        assert users_created_count == len(emails_added)


def test_invalid_args_error_message(root_datasite_client: DatasiteClient) -> None:
    update_args = {
        "name": uuid4().hex,
        "organization": uuid4().hex,
    }

    update = ServerSettingsUpdate(**update_args)

    with pytest.raises(SyftException) as exc:
        root_datasite_client.api.services.settings.update(settings=update)

    assert _SIGNATURE_ERROR_MESSAGE in exc.value.public_message

    with pytest.raises(SyftException) as exc:
        root_datasite_client.api.services.settings.update(update)

    assert _SIGNATURE_ERROR_MESSAGE in exc.value.public_message

    root_datasite_client.api.services.settings.update(**update_args)

    settings = root_datasite_client.api.services.settings.get()
    assert settings.name == update_args["name"]
    assert settings.organization == update_args["organization"]


@pytest.mark.skip(reason="For now notifications can be enabled without credentials.")
def test_notifications_enabled_without_emails_credentials_not_allowed(
    root_datasite_client: DatasiteClient,
) -> None:
    with pytest.raises(SyftException) as exc:
        root_datasite_client.api.services.settings.update(notifications_enabled=True)

    assert _NOTIFICATIONS_ENABLED_WIHOUT_CREDENTIALS_ERROR in exc.value.public_message
