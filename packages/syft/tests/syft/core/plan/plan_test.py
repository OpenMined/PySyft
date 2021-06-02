# stdlib
from typing import Any
from typing import Tuple as TypeTuple

# third party
import pytest
import torch as th

# syft absolute
import syft as sy
from syft import Plan
from syft import make_plan
from syft import serialize
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
from syft.core.plan.plan_builder import ROOT_CLIENT
from syft.core.store.storeable_object import StorableObject
from syft.lib.python.list import List


def test_plan_serialization(client: sy.VirtualMachineClient) -> None:

    # cumbersome way to get a pointer as input for our actions,
    # there is probably a better/shorter way
    t = th.tensor([1, 2, 3])
    tensor_pointer = t.send(client)

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
        map_to_dyn=False,
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
    blob = serialize(plan)
    plan_reconstructed = sy.deserialize(blob=blob)

    # test
    assert isinstance(plan_reconstructed, Plan)
    assert all(isinstance(a, Action) for a in plan_reconstructed.actions)


def test_plan_execution(client: sy.VirtualMachineClient) -> None:
    tensor_pointer1 = th.tensor([1, 2, 3]).send(client)
    tensor_pointer2 = th.tensor([4, 5, 6]).send(client)
    tensor_pointer3 = th.tensor([7, 8, 9]).send(client)

    result_tensor_pointer1 = th.tensor([0, 0, 0]).send(client)
    result_tensor_pointer2 = th.tensor([0, 0, 0]).send(client)

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

    plan_pointer = plan.send(client)

    plan_pointer()

    expected_tensor1 = th.tensor([5, 7, 9])
    expected_tensor2 = th.tensor([12, 15, 18])

    assert all(expected_tensor1 == result_tensor_pointer1.get())
    assert all(expected_tensor2 == result_tensor_pointer2.get())


def test_plan_batched_execution(client: sy.VirtualMachineClient) -> None:
    # placeholders for our input
    input_tensor_pointer1 = th.tensor([0, 0]).send(client)
    input_tensor_pointer2 = th.tensor([0, 0]).send(client)

    # tensors in our model
    model_tensor_pointer1 = th.tensor([1, 2]).send(client)
    model_tensor_pointer2 = th.tensor([3, 4]).send(client)

    # placeholders for intermediate results
    result_tensor_pointer1 = th.tensor([0, 0]).send(client)
    result_tensor_pointer2 = th.tensor([0, 0]).send(client)
    result_tensor_pointer3 = th.tensor([0, 0]).send(client)

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

    plan = Plan(
        [a1, a2, a3], inputs={"x": input_tensor_pointer1, "y": input_tensor_pointer2}
    )
    plan_pointer = plan.send(client)

    # Test
    # x is random input, y is the expected model(x)
    x_batches = [(th.tensor([1, 1]) + i).send(client) for i in range(2)]
    y_batches = [
        ((th.tensor([1, 1]) + i) * th.tensor([1, 2]) + th.tensor([3, 4])).send(client)
        for i in range(2)
    ]

    for x, y in zip(x_batches, y_batches):
        plan_pointer(x=x, y=y)

        # checks if (model(x) == y) == [True, True]
        assert all(result_tensor_pointer3.get(delete_obj=False))


def test_make_plan(client: sy.VirtualMachineClient) -> None:
    @make_plan
    def add_plan(inp=th.zeros((3))) -> th.Tensor:  # type: ignore
        return inp + inp

    input_tensor = th.tensor([1, 2, 3])
    plan_pointer = add_plan.send(client)
    res = plan_pointer(inp=input_tensor)
    assert th.equal(res.get()[0], th.tensor([2, 4, 6]))


@pytest.mark.xfail
def test_plan_deterministic_bytes(root_client: sy.VirtualMachineClient) -> None:
    # TODO: https://github.com/OpenMined/PySyft/issues/5292
    @make_plan
    def add_plan(inp=th.zeros((3))) -> th.Tensor:  # type: ignore
        return inp + inp

    @make_plan
    def add_plan2(inp=th.zeros((3))) -> th.Tensor:  # type: ignore
        return inp + inp

    plan_pointer = add_plan.send(root_client)
    plan2_pointer = add_plan2.send(root_client)

    plan1 = serialize(plan_pointer, to_bytes=True)
    plan2 = serialize(plan2_pointer, to_bytes=True)

    assert plan1 == plan2


def test_make_plan2(root_client: sy.VirtualMachineClient) -> None:
    @make_plan
    def mul_plan(inp=th.zeros((3)), inp2=th.zeros((3))) -> th.Tensor:  # type: ignore
        return inp * inp2

    t1, t2 = th.tensor([1, 2, 3]), th.tensor([1, 2, 3])
    plan_pointer = mul_plan.send(root_client)
    res = plan_pointer(inp=t1, inp2=t2)
    assert th.equal(res.get()[0], th.tensor([1, 4, 9]))


def test_make_plan_error_no_kwargs() -> None:
    def assertRaises(exc, obj, methodname, *args) -> None:  # type: ignore
        with pytest.raises(exc) as e_info:
            getattr(obj, methodname)(*args)
        assert str(e_info) != ""

    def test_define_plan():  # type: ignore
        @make_plan
        def add_plan(inp):  # type: ignore
            return inp + inp

    # we can only define a plan with *kwargs* when using the @make_plan decorator, not with args
    assertRaises(ValueError, test_define_plan, "__call__")


@pytest.mark.slow
def test_mlp_plan(client: sy.VirtualMachineClient) -> None:
    class MLP(sy.Module):
        def __init__(self, torch_ref):  # type: ignore
            super().__init__(torch_ref=torch_ref)
            self.l1 = self.torch_ref.nn.Linear(784, 100)
            self.a1 = self.torch_ref.nn.ReLU()
            self.l2 = self.torch_ref.nn.Linear(100, 10)

        def forward(self, x):  # type: ignore
            x_reshaped = x.view(-1, 28 * 28)
            l1_out = self.a1(self.l1(x_reshaped))
            l2_out = self.l2(l1_out)
            return l2_out

    def set_params(model, params):  # type: ignore
        """happens outside of plan"""
        for p, p_new in zip(model.parameters(), params):
            p.data = p_new.data

    def cross_entropy_loss(logits, targets, batch_size):  # type: ignore
        norm_logits = logits - logits.max()
        log_probs = norm_logits - norm_logits.exp().sum(dim=1, keepdim=True).log()
        return -(targets * log_probs).sum() / batch_size

    def sgd_step(model, lr=0.1):  # type: ignore
        with ROOT_CLIENT.torch.no_grad():
            for p in model.parameters():
                p.data = p.data - lr * p.grad
                # Todo: fix this
                p.grad = th.zeros_like(p.grad.get())

    local_model = MLP(th)  # type: ignore

    @make_plan
    def train(  # type: ignore
        xs=th.rand([64 * 3, 1, 28, 28]),
        ys=th.randint(0, 10, [64 * 3, 10]),
        params=List(local_model.parameters()),
    ):

        model = local_model.send(ROOT_CLIENT)
        set_params(model, params)
        for i in range(2):
            indices = th.tensor(range(64 * i, 64 * (i + 1)))
            x, y = xs.index_select(0, indices), ys.index_select(0, indices)
            out = model(x)
            loss = cross_entropy_loss(out, y, 64)
            loss.backward()
            sgd_step(model)

        return model.parameters()

    train_ptr = train.send(client)

    old_params = local_model.parameters().copy()

    res_ptr = train_ptr(
        xs=th.rand([64 * 3, 1, 28, 28]),
        ys=th.randint(0, 10, [64 * 3, 10]),
        params=local_model.parameters(),
    )

    (new_params,) = res_ptr.get()

    assert not (old_params[0] == new_params[0]).all()


def test_check_placeholder() -> None:
    def test_define_plan() -> None:
        @make_plan
        def add_plan(inp: Any) -> Any:
            return inp + inp

    def assertRaises(
        exc: Exception, obj: object, methodname: str, *args: TypeTuple[Any, ...]
    ) -> None:
        with pytest.raises(exc) as e_info:
            getattr(obj, methodname)(*args)
        assert str(e_info) != ""

    assertRaises(ValueError, test_define_plan, "__call__")  # type: ignore
