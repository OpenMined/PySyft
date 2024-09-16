# third party
from faker import Faker

# syft absolute
import syft
from syft.client.client import SyftClient
from syft.server.worker import Worker
from syft.service.action.action_object import ActionObject
from syft.service.request.request import Request
from syft.service.response import SyftError
from syft.service.response import SyftSuccess


def test_set_tags_delete_requests(faker: Faker, worker: Worker, ds_client: SyftClient):
    """ "
    Scneario: DS client submits a code request. Root client sets some wrong tags, then
    delete the request. DS client then submit the request again, root client then set
    the correct tags.
    """
    root_client: SyftClient = worker.root_client
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
    assert not isinstance(result, SyftError)

    request = root_client.requests.get_all()[0]
    set_tag_res = root_client.api.services.request.set_tags(request, ["tag1", "tag2"])
    assert isinstance(set_tag_res, Request)
    assert set_tag_res.tags == ["tag1", "tag2"]

    del_res = root_client.api.services.request.delete_by_uid(request.id)
    assert isinstance(del_res, SyftSuccess)
    assert len(root_client.api.services.request.get_all()) == 0
    assert root_client.api.services.request.get_by_uid(request.id) is None

    result = ds_client.code.request_code_execution(simple_function)
    assert not isinstance(result, SyftError)
    request = root_client.requests.get_all()[0]
    set_tag_res = root_client.api.services.request.set_tags(
        request, ["computing", "sum"]
    )
    assert isinstance(set_tag_res, Request)
    assert set_tag_res.tags == ["computing", "sum"]
