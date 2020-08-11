import syft as sy
from syft.core.io.location import SpecificLocation
from syft.core.io.address import Address
from syft.core.common.uid import UID
from syft.core.node.common.action.save_object_action import SaveObjectAction

import torch as th


def test_save_object_action_serde():

    uid = UID()
    obj = th.tensor([1, 2, 3])
    addr = Address(network=SpecificLocation(), device=SpecificLocation())

    msg = SaveObjectAction(obj_id=uid, obj=obj, address=addr)

    blob = msg.serialize()

    msg2 = sy.deserialize(blob=blob)

    assert msg2.obj_id == msg.obj_id
    assert (msg2.obj == msg.obj).all()
    assert msg2.obj.id == msg.obj.id
    assert msg2.address == msg.address
