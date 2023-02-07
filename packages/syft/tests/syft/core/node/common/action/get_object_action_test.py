# syft absolute
import syft as sy
from syft import serialize
from syft.core.common.uid import UID
from syft.core.node.common.action.get_object_action import GetObjectAction


def test_get_object_action_serde() -> None:
    msg = GetObjectAction(
        id_at_location=UID(), address=UID(), reply_to=UID(), msg_id=UID()
    )
    blob = serialize(msg)
    msg2 = sy.deserialize(blob=blob)

    assert msg.id == msg2.id
    assert msg.id_at_location == msg2.id_at_location
    assert msg.address == msg2.address
    assert msg.reply_to == msg2.reply_to
