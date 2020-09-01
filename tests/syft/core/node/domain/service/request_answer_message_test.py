# syft absolute
from syft import deserialize
from syft import serialize
from syft.core.common import UID
from syft.core.io.address import Address
from syft.core.node.domain.service import RequestAnswerMessage
from syft.core.node.domain.service import RequestAnswerResponse
from syft.core.node.domain.service import RequestStatus


def test_request_answer_message() -> None:

    addr = Address()

    msg = RequestAnswerMessage(request_id=UID(), address=addr, reply_to=addr)

    serialized = serialize(obj=msg)
    new_msg = deserialize(blob=serialized)

    assert msg.request_id == new_msg.request_id
    assert msg.address == new_msg.address
    assert msg.reply_to == new_msg.reply_to


def test_request_answer_response() -> None:

    addr = Address()

    msg = RequestAnswerResponse(
        request_id=UID(), address=addr, status=RequestStatus.Pending
    )

    serialized = serialize(obj=msg)
    new_msg = deserialize(blob=serialized)

    assert msg.request_id == new_msg.request_id
    assert msg.address == new_msg.address
    assert msg.status == new_msg.status
