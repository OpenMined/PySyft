# third party
import torch as th

# syft absolute
import syft as sy
from syft.core.common.uid import UID
from syft.core.io.address import Address
from syft.core.io.location import SpecificLocation
from syft.core.node.common.action.save_object_action import SaveObjectAction


def test_save_object_action_serde() -> None:

    uid = UID()
    obj = th.tensor([1, 2, 3])
    addr = Address(network=SpecificLocation(), device=SpecificLocation())

    msg = SaveObjectAction(id_at_location=uid, obj=obj, address=addr)

    blob = msg.serialize()

    msg2 = sy.deserialize(blob=blob)

    assert msg2.id_at_location == msg.id_at_location
    assert (msg2.obj == msg.obj).all()
    # Tensors do not automatically get IDs anymore
    # assert msg2.obj.id == msg.obj.id
    assert msg2.address == msg.address
