# syft absolute
from syft import deserialize
from syft import serialize
from syft.core.common import UID
from syft.core.node.common.node_service.request_answer.request_answer_messages import (
    RequestAnswerMessage,
)
from syft.core.node.common.node_service.request_answer.request_answer_messages import (
    RequestAnswerResponse,
)
from syft.core.node.common.node_service.request_receiver.request_receiver_messages import (
    RequestStatus,
)


def test_request_answer_message() -> None:

    addr = UID()

    msg = RequestAnswerMessage(request_id=UID(), address=addr, reply_to=addr)

    serialized = serialize(obj=msg)
    new_msg = deserialize(blob=serialized)

    assert msg.request_id == new_msg.request_id
    assert msg.address == new_msg.address
    assert msg.reply_to == new_msg.reply_to


def test_request_answer_response() -> None:

    addr = UID()

    msg = RequestAnswerResponse(
        request_id=UID(), address=addr, status=RequestStatus.Pending
    )

    serialized = serialize(obj=msg)
    new_msg = deserialize(blob=serialized)

    assert msg.request_id == new_msg.request_id
    assert msg.address == new_msg.address
    assert msg.status == new_msg.status
