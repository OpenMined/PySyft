import syft as sy
import torch as th
import sys

from syft.core.node.common.action.get_object_action import GetObjectAction
from syft.core.node.common.action.function_or_constructor_action import RunFunctionOrConstructorAction
from syft.core.node.common.action.run_class_method_action import RunClassMethodAction
from syft.core.node.common.action.garbage_collect_object_action import GarbageCollectObjectAction
from syft.core.node.common.action.get_enum_attribute_action import EnumAttributeAction
from syft.core.node.common.action.get_or_set_property_action import GetOrSetPropertyAction, PropertyActions
from syft.core.node.common.action.get_or_set_static_attribute_action import GetSetStaticAttributeAction, StaticAttributeAction
from syft.core.node.common.action.save_object_action import SaveObjectAction
from syft.core.store.storeable_object import StorableObject


from syft.core.node.common.action.common import Action
from syft.proto.core.node.common.plan.plan_pb2 import Plan as Plan_PB

from typing import List
from syft.core.common.uid import UID
from syft.core.io.address import Address
from syft.core.common.serde.deserialize import _deserialize
from syft.core.common.object import Serializable
from syft.proto.core.node.common.action.action_pb2 import Action as Action_PB
from syft.core.node.common.plan.plan import Plan
from syft.core.pointer.pointer import Pointer


def test_plan_serialization() -> None:

    # cumbersome way to get a pointer as input for our actions,
    # there is probably a better/shorter way
    alice = sy.VirtualMachine(name="alice")
    alice_client = alice.get_client()
    t = th.tensor([1,2,3])
    tensor_pointer = t.send(alice_client)

    #define actions
    a1 = GetObjectAction(id_at_location=UID(), address=Address(), reply_to=Address(), msg_id=UID())
    a2 = RunFunctionOrConstructorAction(path="torch.Tensor.add", args=tuple(), kwargs={}, id_at_location=UID(),
                                        address=Address(),msg_id=UID())

    a3 = RunClassMethodAction(path="torch.Tensor.add", _self=tensor_pointer,
                            args=[], kwargs={}, id_at_location=UID(),
                            address=Address(),msg_id=UID())

    a4 = GarbageCollectObjectAction(id_at_location=UID(), address=Address())
    a5 = EnumAttributeAction(path="", id_at_location=UID(), address=Address())

    a6 = GetOrSetPropertyAction(path="", _self=tensor_pointer, id_at_location=UID(), address=Address(), args=[],
                                kwargs={}, action=PropertyActions.GET)
    a7 = GetSetStaticAttributeAction(path="", id_at_location=UID(),address=Address(),
                                    action=StaticAttributeAction.GET)
    a8 = SaveObjectAction(id_at_location=UID(), obj=StorableObject(id=UID(), data=t), address=Address())

    # define plan
    plan = Plan([a1,a2, a3, a4, a5, a6, a7, a8])

    # serialize / deserialize
    blob = plan.serialize()
    plan_reconstructed = sy.deserialize(blob=blob)

    # test
    assert isinstance(plan_reconstructed, Plan)
    assert all(isinstance(a, Action) for a in plan_reconstructed.actions)


def test_plan_execution() -> None:
    alice = sy.VirtualMachine(name="alice")
    alice_client = alice.get_client()

    tensor_pointer1 = th.tensor([1,2,3]).send(alice_client)
    tensor_pointer2 = th.tensor([4,5,6]).send(alice_client)
    tensor_pointer3 = th.tensor([7,8,9]).send(alice_client)

    result_tensor_pointer1 = th.tensor([0,0,0]).send(alice_client)
    result_tensor_pointer2 = th.tensor([0,0,0]).send(alice_client)

    result1_uid = result_tensor_pointer1.id_at_location
    result2_uid = result_tensor_pointer2.id_at_location

    a1 = RunClassMethodAction(path="torch.Tensor.add", _self=tensor_pointer1, args=[tensor_pointer2], kwargs={},
                            id_at_location=result1_uid, address=Address(),msg_id=UID())

    a2 = RunClassMethodAction(path="torch.Tensor.add", _self=result_tensor_pointer1, args=[tensor_pointer3],
                            kwargs={}, id_at_location=result2_uid, address=Address(),msg_id=UID())

    plan = Plan([a1,a2])

    plan_pointer = plan.send(alice_client)

    plan_pointer.execute();

    expected_tensor1 = th.tensor([5,7,9])
    expected_tensor2 = th.tensor([12,15,18])

    assert(all(expected_tensor1 == result_tensor_pointer1.get()))
    assert(all(expected_tensor2 == result_tensor_pointer2.get()))
