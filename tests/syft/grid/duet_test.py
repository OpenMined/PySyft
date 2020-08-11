import syft as sy
from syft.grid.duet.request import RequestResponse, RequestMessage, RequestStatus
from syft.core.common import UID
import torch as th


def test_duet():
    duet = sy.Duet(host="127.0.0.1", port=5000)
    obj_id = duet.id
    duet.stop()
    assert "ObjectWithID" in f"{obj_id}"


def test_request_message_creation():
    obj = RequestMessage(
        request_name="request", request_description="request description"
    )

    assert obj.request_name == "request"
    assert obj.request_description == "request description"


def test_request_message_serde():
    obj = RequestMessage(
        request_name="request", request_description="request description"
    )
    serialized_request = sy.serialize(obj=obj)
    new_obj = sy.deserialize(blob=serialized_request)

    assert obj.request_name == new_obj.request_name
    assert obj.request_description == new_obj.request_description
    assert obj.request_id == new_obj.request_id


def test_request_response():
    id = UID()
    status = RequestStatus.Pending
    obj = RequestResponse(status=status, request_id=id)

    assert obj.status == status
    assert obj.request_id == id


def test_request_response_serde():
    obj = RequestResponse(status=RequestStatus.Pending, request_id=UID())

    serialized_obj = sy.serialize(obj=obj)
    new_obj = sy.deserialize(blob=serialized_obj)

    assert obj.status == new_obj.status
    assert obj.request_id == new_obj.request_id


def test_duet_send():
    duet = sy.Duet(host="127.0.0.1", port=5001)

    x = th.tensor([1, 2, 3])
    xp = x.send(duet)

    assert xp.id_at_location == x.id

    duet.stop()
