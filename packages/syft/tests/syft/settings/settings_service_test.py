# stdlib
from copy import deepcopy
from datetime import datetime
from unittest import mock

# third party
from faker import Faker
from pytest import MonkeyPatch
from result import Err
from result import Ok

# syft absolute
import syft
from syft.abstract_server import ServerSideType
from syft.server.credentials import SyftSigningKey
from syft.server.credentials import SyftVerifyKey
from syft.service.context import AuthedServiceContext
from syft.service.notifier.notifier import NotifierSettings
from syft.service.notifier.notifier_stash import NotifierStash
from syft.service.response import SyftError
from syft.service.response import SyftSuccess
from syft.service.settings.settings import ServerSettings
from syft.service.settings.settings import ServerSettingsUpdate
from syft.service.settings.settings_service import SettingsService
from syft.service.settings.settings_stash import SettingsStash
from syft.service.user.user import UserCreate
from syft.service.user.user_roles import ServiceRole


def test_settingsservice_get_success(
    monkeypatch: MonkeyPatch,
    settings_service: SettingsService,
    settings: ServerSettings,
    authed_context: AuthedServiceContext,
) -> None:
    mock_stash_get_all_output = [settings, settings]
    expected_output = Ok(mock_stash_get_all_output[0])

    def mock_stash_get_all(credentials) -> Ok:
        return Ok(mock_stash_get_all_output)

    monkeypatch.setattr(settings_service.stash, "get_all", mock_stash_get_all)

    response = settings_service.get(context=authed_context)

    assert isinstance(response.ok(), ServerSettings)
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
    settings: ServerSettings,
    authed_context: AuthedServiceContext,
) -> None:
    response = settings_service.set(authed_context, settings)

    assert response.is_ok() is True
    assert isinstance(response.ok(), ServerSettings)
    assert response.ok() == settings


def test_settingsservice_set_fail(
    monkeypatch: MonkeyPatch,
    settings_service: SettingsService,
    settings: ServerSettings,
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
    settings: ServerSettings,
) -> ServerSettings:
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
    settings: ServerSettings,
    update_settings: ServerSettingsUpdate,
    authed_context: AuthedServiceContext,
    notifier_stash: NotifierStash,
) -> None:
    # add a mock settings to the stash
    mock_settings = add_mock_settings(
        authed_context.credentials, settings_stash, settings
    )

    # get a new settings according to update_settings
    new_settings = deepcopy(settings)
    update_kwargs = update_settings.to_dict(exclude_empty=True).items()
    for field_name, value in update_kwargs:
        setattr(new_settings, field_name, value)

    assert new_settings != settings
    assert new_settings != mock_settings
    assert mock_settings == settings

    mock_stash_get_all_output = [mock_settings, mock_settings]

    def mock_stash_get_all(root_verify_key) -> Ok:
        return Ok(mock_stash_get_all_output)

    monkeypatch.setattr(settings_service.stash, "get_all", mock_stash_get_all)

    # Mock the get_service method to return a mocked notifier_service with the notifier_stash
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

    # not_updated_settings = response.ok()[1]

    assert isinstance(response, SyftSuccess)
    # assert (
    #     not_updated_settings.to_dict() == settings.to_dict()
    # )  # the second settings is not updated


def test_settingsservice_update_stash_get_all_fail(
    monkeypatch: MonkeyPatch,
    settings_service: SettingsService,
    update_settings: ServerSettingsUpdate,
    authed_context: AuthedServiceContext,
) -> None:
    # the stash.get_all() function fails
    mock_error_message = "database failure"

    def mock_stash_get_all_error(credentials) -> Err:
        return Err(mock_error_message)

    monkeypatch.setattr(settings_service.stash, "get_all", mock_stash_get_all_error)
    response = settings_service.update(context=authed_context, settings=update_settings)

    assert isinstance(response, SyftError)
    assert response.message == mock_error_message


def test_settingsservice_update_stash_empty(
    settings_service: SettingsService,
    update_settings: ServerSettingsUpdate,
    authed_context: AuthedServiceContext,
) -> None:
    response = settings_service.update(context=authed_context, settings=update_settings)

    assert isinstance(response, SyftError)
    assert response.message == "No settings found"


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

    def mock_stash_get_all(credentials) -> Ok:
        return Ok(mock_stash_get_all_output)

    monkeypatch.setattr(settings_service.stash, "get_all", mock_stash_get_all)

    mock_update_error_message = "Failed to update obj ServerMetadata"

    def mock_stash_update_error(credentials, update_settings: ServerSettings) -> Err:
        return Err(mock_update_error_message)

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

    response = settings_service.update(context=authed_context, settings=update_settings)

    assert isinstance(response, SyftError)
    assert response.message == mock_update_error_message


def test_settings_allow_guest_registration(
    monkeypatch: MonkeyPatch, faker: Faker
) -> None:
    # Create a new worker

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
        worker = syft.Worker.named(name=faker.name(), reset=True)
        guest_datasite_client = worker.guest_client
        root_datasite_client = worker.root_client

        email1 = faker.email()
        email2 = faker.email()

        response_1 = root_datasite_client.register(
            email=email1, password="joker123", password_verify="joker123", name="Joker"
        )
        assert isinstance(response_1, SyftSuccess)

        # by default, the guest client can't register new user
        response_2 = guest_datasite_client.register(
            email=email2,
            password="harley123",
            password_verify="harley123",
            name="Harley",
        )
        assert isinstance(response_2, SyftError)

        assert any(user.email == email1 for user in root_datasite_client.users)

    # only after the root client enable other users to signup, they can
    mock_server_settings.signup_enabled = True
    with mock.patch(
        "syft.Worker.settings",
        new_callable=mock.PropertyMock,
        return_value=mock_server_settings,
    ):
        worker = syft.Worker.named(name=faker.name(), reset=True)
        guest_datasite_client = worker.guest_client
        root_datasite_client = worker.root_client

        password = faker.email()
        response_3 = guest_datasite_client.register(
            email=email2,
            password=password,
            password_verify=password,
            name=faker.name(),
        )
        assert isinstance(response_3, SyftSuccess)

        assert any(user.email == email2 for user in root_datasite_client.users)


def test_user_register_for_role(monkeypatch: MonkeyPatch, faker: Faker):
    # Mock patch this env variable to remove race conditions
    # where signup is enabled.
    def get_mock_client(faker, root_client, role):
        user_create = UserCreate(
            name=faker.name(),
            email=faker.email(),
            role=role,
            password="password",
            password_verify="password",
        )
        result = root_client.users.create(**user_create)
        assert not isinstance(result, SyftError)

        guest_client = root_client.guest()
        return guest_client.login(
            email=user_create.email, password=user_create.password
        )

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
        worker = syft.Worker.named(name=faker.name(), reset=True)
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
            emails_added.append(email)

        ds_client = get_mock_client(
            faker=faker, root_client=root_client, role=ServiceRole.DATA_SCIENTIST
        )

        response = ds_client.register(
            name=faker.name(),
            email=faker.email(),
            password="password",
            password_verify="password",
        )
        assert isinstance(response, SyftError)

        users_created_count = sum(
            [u.email in emails_added for u in root_client.users.get_all()]
        )
        assert users_created_count == len(emails_added)
