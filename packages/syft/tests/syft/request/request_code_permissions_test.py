# syft absolute
import syft
from syft.client.client import SyftClient
from syft.server.worker import Worker
from syft.service.action.action_object import ActionObject
from syft.service.code.user_code import UserCode
from syft.service.response import SyftSuccess


def test_code_request_submitted_by_admin_only_admin_can_view(
    worker: Worker, ds_client: SyftClient
):
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

    project = syft.Project(name="test", members=[root_client])

    result = project.create_code_request(simple_function, root_client)
    assert isinstance(result, SyftSuccess)

    # only root should be able to see request and access code
    ds_request_all = ds_client.requests.get_all()
    assert len(ds_request_all) == 0

    root_request_all = root_client.requests.get_all()
    assert len(root_request_all) == 1
    root_code_access = root_request_all[0].code
    assert isinstance(root_code_access, UserCode)


def test_code_request_submitted_by_ds_root_and_ds_can_view(
    worker: Worker, ds_client: SyftClient
):
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

    project = syft.Project(name="test", members=[ds_client])

    result = project.create_code_request(simple_function, ds_client)
    assert isinstance(result, SyftSuccess)

    # both root and ds should be able to see request and access code
    ds_request_all = ds_client.requests.get_all()
    assert len(ds_request_all) == 1
    ds_code_access = ds_request_all[0].code
    assert isinstance(ds_code_access, UserCode)

    root_request_all = root_client.requests.get_all()
    assert len(root_request_all) == 1
    root_code_access = root_request_all[0].code
    assert isinstance(root_code_access, UserCode)
