# syft absolute
import syft as sy
from syft.core.common.uid import UID
from syft.core.io.address import Address
from syft.core.io.location import SpecificLocation
from syft.core.node.common.action.garbage_collect_object_action import (
    GarbageCollectObjectAction,
)


def test_garbage_collection_object_action_serde() -> None:
    uid = UID()
    addr = Address(network=SpecificLocation(), device=SpecificLocation())

    msg = GarbageCollectObjectAction(obj_id=uid, address=addr)

    blob = msg.serialize()

    msg2 = sy.deserialize(blob=blob)

    assert msg2.obj_id == msg.obj_id
    assert msg2.address == msg.address
