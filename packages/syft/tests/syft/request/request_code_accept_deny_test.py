# third party
import pytest

# syft absolute
import syft
from syft.client.client import SyftClient
from syft.server.worker import Worker
from syft.service.action.action_object import ActionObject
from syft.service.action.action_permissions import ActionPermission
from syft.service.code.user_code import UserCode
from syft.service.code.user_code import UserCodeStatus
from syft.service.context import ChangeContext
from syft.service.request.request import ActionStoreChange
from syft.service.request.request import ObjectMutation
from syft.service.request.request import RequestStatus
from syft.service.request.request import UserCodeStatusChange
from syft.service.response import SyftSuccess
from syft.service.settings.settings_service import SettingsService
from syft.store.linked_obj import LinkedObject
from syft.types.errors import SyftException


def test_object_mutation(worker: Worker):
    root_client = worker.root_client
    setting = root_client.api.services.settings.get()
    linked_obj = LinkedObject.from_obj(setting, SettingsService, server_uid=worker.id)
    original_name = setting.organization
    new_name = "Test Organization"

    object_mutation = ObjectMutation(
        linked_obj=linked_obj,
        attr_name="organization",
        match_type=True,
        value=new_name,
    )

    change_context = ChangeContext(
        server=worker,
        approving_user_credentials=root_client.credentials.verify_key,
    )

    result = object_mutation.apply(change_context)

    assert result.is_ok()

    setting = root_client.api.services.settings.get()

    assert setting.organization == new_name

    object_mutation.undo(context=change_context)

    setting = root_client.api.services.settings.get()

    assert setting.organization == original_name


def test_action_store_change(worker: Worker, ds_client: SyftClient):
    root_client = worker.root_client
    dummy_data = [1, 2, 3]
    data = ActionObject.from_obj(dummy_data)
    action_obj = data.send(root_client)

    assert action_obj.get() == dummy_data

    action_object_link = LinkedObject.from_obj(
        action_obj, server_uid=action_obj.syft_server_uid
    )
    permission_change = ActionStoreChange(
        linked_obj=action_object_link,
        apply_permission_type=ActionPermission.READ,
    )

    change_context = ChangeContext(
        server=worker,
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

    with pytest.raises(SyftException) as exc:
        action_obj_ptr.get()

    assert exc.type is SyftException
    assert "Permission", "denied" in exc.value.public_message


def test_user_code_status_change(worker: Worker, ds_client: SyftClient):
    root_client = worker.root_client
    dummy_data = [1, 2, 3]
    data = ActionObject.from_obj(dummy_data)
    action_obj = data.send(root_client)

    @syft.syft_function(
        input_policy=syft.ExactMatch(data=action_obj),
        output_policy=syft.SingleExecutionExactOutput(),
    )
    def simple_function(data):
        return sum(data)

    result = ds_client.code.submit(simple_function)
    assert isinstance(result, SyftSuccess)

    user_code: UserCode = ds_client.code.get_all()[0]

    linked_user_code = LinkedObject.from_obj(user_code, server_uid=worker.id)

    user_code_change = UserCodeStatusChange(
        value=UserCodeStatus.APPROVED,
        linked_user_code=linked_user_code,
        linked_obj=user_code.status_link,
    )

    change_context = ChangeContext(
        server=worker,
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


def test_code_accept_deny(worker: Worker, ds_client: SyftClient):
    root_client = worker.root_client
    dummy_data = [1, 2, 3]
    data = ActionObject.from_obj(dummy_data)
    action_obj = data.send(root_client)

    @syft.syft_function(
        input_policy=syft.ExactMatch(data=action_obj),
        output_policy=syft.SingleExecutionExactOutput(),
    )
    def simple_function(data):
        return sum(data)

    result = ds_client.code.request_code_execution(simple_function)
    request = root_client.requests.get_all()[0]
    result = request.approve()
    assert isinstance(result, SyftSuccess)

    request = root_client.requests.get_all()[0]
    assert request.status == RequestStatus.APPROVED

    result = ds_client.code.simple_function(data=action_obj)
    assert result.get() == sum(dummy_data)

    deny_reason = "Function output needs differential privacy!!"
    result = request.deny(reason=deny_reason)
    assert isinstance(result, SyftSuccess)

    request = root_client.requests.get_all()[0]
    assert request.status == RequestStatus.REJECTED

    user_code = ds_client.code.get_all()[0]
    assert not user_code.status.approved

    with pytest.raises(SyftException) as exc:
        ds_client.code.simple_function(data=action_obj)

    assert exc.type is SyftException
    assert deny_reason in exc.value.public_message
