# syft absolute
import syft as sy
from syft.core.common.uid import UID
from syft.core.io.address import Address
from syft.core.node.common.action.get_object_action import GetObjectAction


def test_get_object_action_serde() -> None:
    msg = GetObjectAction(
        obj_id=UID(), address=Address(), reply_to=Address(), msg_id=UID()
    )
    blob = msg.serialize()
    msg2 = sy.deserialize(blob=blob)

    assert msg.id == msg2.id
    assert msg.obj_id == msg2.obj_id
    assert msg.address == msg2.address
    assert msg.reply_to == msg2.reply_to
