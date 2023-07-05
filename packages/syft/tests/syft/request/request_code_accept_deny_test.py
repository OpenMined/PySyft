# stdlib
from textwrap import dedent

# third party
from faker import Faker
import pytest

# syft absolute
import syft
from syft.client.client import SyftClient
from syft.node.worker import Worker
from syft.service.action.action_object import ActionObject
from syft.service.action.action_permissions import ActionPermission
from syft.service.code.user_code import UserCodeStatus
from syft.service.context import ChangeContext
from syft.service.request.request import ActionStoreChange
from syft.service.request.request import ObjectMutation
from syft.service.request.request import RequestStatus
from syft.service.request.request import UserCodeStatusChange
from syft.service.request.request_service import RequestService
from syft.service.response import SyftError
from syft.service.response import SyftSuccess
from syft.service.settings.settings_service import SettingsService
from syft.store.document_store import DocumentStore
from syft.store.linked_obj import LinkedObject


@pytest.fixture
def request_service(document_store: DocumentStore):
    return RequestService(store=document_store)


def get_ds_client(faker: Faker, root_client: SyftClient, guest_client: SyftClient):
    guest_email = faker.email()
    password = "mysecretpassword"
    result = root_client.register(
        name=faker.name(), email=guest_email, password=password
    )
    assert isinstance(result, SyftSuccess)
    guest_client.login(email=guest_email, password=password)
    return guest_client


def test_object_mutation(worker: Worker):
    root_client = worker.root_client
    setting = root_client.api.services.settings.get()
    linked_obj = LinkedObject.from_obj(setting, SettingsService, node_uid=worker.id)
    original_name = setting.organization
    new_name = "Test Organization"

    object_mutation = ObjectMutation(
        linked_obj=linked_obj,
        attr_name="organization",
        match_type=True,
        value=new_name,
    )

    change_context = ChangeContext(
        node=worker,
        approving_user_credentials=root_client.credentials.verify_key,
    )

    result = object_mutation.apply(change_context)

    assert isinstance(result, SyftSuccess)

    setting = root_client.api.services.settings.get()

    assert setting.organization == new_name

    object_mutation.undo(context=change_context)

    setting = root_client.api.services.settings.get()

    assert setting.organization == original_name


def test_action_store_change(faker: Faker, worker: Worker):
    root_client = worker.root_client
    dummy_data = [1, 2, 3]
    data = ActionObject.from_obj(dummy_data)
    action_obj = root_client.api.services.action.set(data)

    assert action_obj.get() == dummy_data

    ds_client = get_ds_client(faker, root_client, worker.guest_client)

    action_object_link = LinkedObject.from_obj(
        action_obj, node_uid=action_obj.syft_node_uid
    )
    permission_change = ActionStoreChange(
        linked_obj=action_object_link,
        apply_permission_type=ActionPermission.READ,
    )

    change_context = ChangeContext(
        node=worker,
        approving_user_credentials=root_client.credentials.verify_key,
        requesting_user_credentials=ds_client.credentials.verify_key,
    )

    result = permission_change.apply(change_context)

    assert result.is_ok()

    action_obj_ptr = ds_client.api.services.action.get_pointer(action_obj.id)

    result = action_obj_ptr.get()
    assert result == dummy_data

    result = permission_change.undo(change_context)
    assert result.is_ok()

    result = action_obj_ptr.get()
    assert result.is_err()


def test_user_code_status_change(faker: Faker, worker: Worker):
    root_client = worker.root_client
    dummy_data = [1, 2, 3]
    data = ActionObject.from_obj(dummy_data)
    action_obj = root_client.api.services.action.set(data)

    ds_client = get_ds_client(faker, root_client, worker.guest_client)

    @syft.syft_function(
        input_policy=syft.ExactMatch(data=action_obj),
        output_policy=syft.SingleExecutionExactOutput(),
    )
    def simple_function(data):
        return sum(data)

    simple_function.code = dedent(simple_function.code)
    result = ds_client.code.submit(simple_function)
    assert isinstance(result, SyftSuccess)

    user_code = ds_client.code.get_all()[0]

    linked_obj = LinkedObject.from_obj(user_code, node_uid=worker.id)

    user_code_change = UserCodeStatusChange(
        value=UserCodeStatus.EXECUTE, linked_obj=linked_obj
    )

    change_context = ChangeContext(
        node=worker,
        approving_user_credentials=root_client.credentials.verify_key,
        requesting_user_credentials=ds_client.credentials.verify_key,
    )

    result = user_code_change.apply(change_context)

    user_code = ds_client.code.get_all()[0]

    assert user_code.status.approved

    result = user_code_change.undo(change_context)
    assert result.is_ok()

    user_code = ds_client.code.get_all()[0]

    assert not user_code.status.approved


def test_code_accept_deny(faker: Faker, worker: Worker):
    root_client = worker.root_client
    dummy_data = [1, 2, 3]
    data = ActionObject.from_obj(dummy_data)
    action_obj = root_client.api.services.action.set(data)

    ds_client = get_ds_client(faker, root_client, worker.guest_client)

    @syft.syft_function(
        input_policy=syft.ExactMatch(data=action_obj),
        output_policy=syft.SingleExecutionExactOutput(),
    )
    def simple_function(data):
        return sum(data)

    simple_function.code = dedent(simple_function.code)

    result = ds_client.code.request_code_execution(simple_function)
    assert not isinstance(result, SyftError)

    request = root_client.requests.get_all()[0]
    result = request.accept_by_depositing_result(result=10)
    assert isinstance(result, SyftSuccess)

    request = root_client.requests.get_all()[0]
    assert request.status == RequestStatus.APPROVED
    result = ds_client.code.simple_function(data=action_obj)
    assert result.get() == 10

    result = request.deny(reason="Function output needs differential privacy !!")
    assert isinstance(result, SyftSuccess)

    request = root_client.requests.get_all()[0]
    assert request.status == RequestStatus.REJECTED

    user_code = ds_client.code.get_all()[0]
    assert not user_code.status.approved

    result = ds_client.code.simple_function(data=action_obj)
    assert isinstance(result, SyftError)
    assert "UserCodeStatus.DENIED" in result.message
