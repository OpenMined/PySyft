# third party
import torch as th

# syft absolute
import syft as sy
from syft.core.common.uid import UID
from syft.core.io.address import Address
from syft.core.node.common.action.common import Action
from syft.core.node.common.action.function_or_constructor_action import (
    RunFunctionOrConstructorAction,
)
from syft.core.node.common.action.garbage_collect_object_action import (
    GarbageCollectObjectAction,
)
from syft.core.node.common.action.get_enum_attribute_action import EnumAttributeAction
from syft.core.node.common.action.get_object_action import GetObjectAction
from syft.core.node.common.action.get_or_set_property_action import (
    GetOrSetPropertyAction,
)
from syft.core.node.common.action.get_or_set_property_action import PropertyActions
from syft.core.node.common.action.get_or_set_static_attribute_action import (
    GetSetStaticAttributeAction,
)
from syft.core.node.common.action.get_or_set_static_attribute_action import (
    StaticAttributeAction,
)
from syft.core.node.common.action.run_class_method_action import RunClassMethodAction
from syft.core.node.common.action.save_object_action import SaveObjectAction
from syft.core.node.common.plan.plan import Plan
from syft.core.store.storeable_object import StorableObject


def test_plan_serialization() -> None:

    # cumbersome way to get a pointer as input for our actions,
    # there is probably a better/shorter way
    alice = sy.VirtualMachine(name="alice")
    alice_client = alice.get_client()
    t = th.tensor([1, 2, 3])
    tensor_pointer = t.send(alice_client)

    # define actions
    a1 = GetObjectAction(
        id_at_location=UID(), address=Address(), reply_to=Address(), msg_id=UID()
    )
    a2 = RunFunctionOrConstructorAction(
        path="torch.Tensor.add",
        args=tuple(),
        kwargs={},
        id_at_location=UID(),
        address=Address(),
        msg_id=UID(),
    )

    a3 = RunClassMethodAction(
        path="torch.Tensor.add",
        _self=tensor_pointer,
        args=[],
        kwargs={},
        id_at_location=UID(),
        address=Address(),
        msg_id=UID(),
    )

    a4 = GarbageCollectObjectAction(id_at_location=UID(), address=Address())
    a5 = EnumAttributeAction(path="", id_at_location=UID(), address=Address())

    a6 = GetOrSetPropertyAction(
        path="",
        _self=tensor_pointer,
        id_at_location=UID(),
        address=Address(),
        args=[],
        kwargs={},
        action=PropertyActions.GET,
    )
    a7 = GetSetStaticAttributeAction(
        path="",
        id_at_location=UID(),
        address=Address(),
        action=StaticAttributeAction.GET,
    )
    a8 = SaveObjectAction(obj=StorableObject(id=UID(), data=t), address=Address())

    # define plan
    plan = Plan([a1, a2, a3, a4, a5, a6, a7, a8])

    # serialize / deserialize
    blob = plan.serialize()
    plan_reconstructed = sy.deserialize(blob=blob)

    # test
    assert isinstance(plan_reconstructed, Plan)
    assert all(isinstance(a, Action) for a in plan_reconstructed.actions)


def test_plan_execution() -> None:
    alice = sy.VirtualMachine(name="alice")
    alice_client = alice.get_client()

    tensor_pointer1 = th.tensor([1, 2, 3]).send(alice_client)
    tensor_pointer2 = th.tensor([4, 5, 6]).send(alice_client)
    tensor_pointer3 = th.tensor([7, 8, 9]).send(alice_client)

    result_tensor_pointer1 = th.tensor([0, 0, 0]).send(alice_client)
    result_tensor_pointer2 = th.tensor([0, 0, 0]).send(alice_client)

    result1_uid = result_tensor_pointer1.id_at_location
    result2_uid = result_tensor_pointer2.id_at_location

    a1 = RunClassMethodAction(
        path="torch.Tensor.add",
        _self=tensor_pointer1,
        args=[tensor_pointer2],
        kwargs={},
        id_at_location=result1_uid,
        address=Address(),
        msg_id=UID(),
    )

    a2 = RunClassMethodAction(
        path="torch.Tensor.add",
        _self=result_tensor_pointer1,
        args=[tensor_pointer3],
        kwargs={},
        id_at_location=result2_uid,
        address=Address(),
        msg_id=UID(),
    )

    plan = Plan([a1, a2])

    plan_pointer = plan.send(alice_client)

    plan_pointer.execute()

    expected_tensor1 = th.tensor([5, 7, 9])
    expected_tensor2 = th.tensor([12, 15, 18])

    assert all(expected_tensor1 == result_tensor_pointer1.get())
    assert all(expected_tensor2 == result_tensor_pointer2.get())


def test_plan_batched_execution() -> None:
    alice = sy.VirtualMachine(name="alice")
    alice_client = alice.get_client()

    # placeholders for our input
    input_tensor_pointer1 = th.tensor([0, 0]).send(alice_client)
    input_tensor_pointer2 = th.tensor([0, 0]).send(alice_client)

    # tensors in our model
    model_tensor_pointer1 = th.tensor([1, 2]).send(alice_client)
    model_tensor_pointer2 = th.tensor([3, 4]).send(alice_client)

    # placeholders for intermediate results
    result_tensor_pointer1 = th.tensor([0, 0]).send(alice_client)
    result_tensor_pointer2 = th.tensor([0, 0]).send(alice_client)
    result_tensor_pointer3 = th.tensor([0, 0]).send(alice_client)

    # define plan
    a1 = RunClassMethodAction(
        path="torch.Tensor.mul",
        _self=input_tensor_pointer1,
        args=[model_tensor_pointer1],
        kwargs={},
        id_at_location=result_tensor_pointer1.id_at_location,
        address=Address(),
        msg_id=UID(),
    )

    a2 = RunClassMethodAction(
        path="torch.Tensor.add",
        _self=result_tensor_pointer1,
        args=[model_tensor_pointer2],
        kwargs={},
        id_at_location=result_tensor_pointer2.id_at_location,
        address=Address(),
        msg_id=UID(),
    )

    a3 = RunFunctionOrConstructorAction(
        path="torch.eq",
        args=[result_tensor_pointer2, input_tensor_pointer2],
        kwargs={},
        id_at_location=result_tensor_pointer3.id_at_location,
        address=Address(),
        msg_id=UID(),
    )

    plan = Plan([a1, a2, a3], inputs=[input_tensor_pointer1, input_tensor_pointer2])
    plan_pointer = plan.send(alice_client)

    # Test
    # x is random input, y is the expected model(x)
    x_batches = [(th.tensor([1, 1]) + i).send(alice_client) for i in range(2)]
    y_batches = [
        ((th.tensor([1, 1]) + i) * th.tensor([1, 2]) + th.tensor([3, 4])).send(
            alice_client
        )
        for i in range(2)
    ]

    for x, y in zip(x_batches, y_batches):
        plan_pointer.execute(x, y)

        # checks if (model(x) == y) == [True, True]
        assert all(result_tensor_pointer3.get())
