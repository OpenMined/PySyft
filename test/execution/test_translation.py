import pytest

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import syft as sy
from itertools import starmap
from syft.execution.placeholder import PlaceHolder
from syft.execution.plan import Plan
from syft.execution.translation.torchscript import PlanTranslatorTorchscript
from syft.serde.serde import deserialize
from syft.serde.serde import serialize


def test_plan_can_be_jit_traced(hook, workers):
    args_shape = [(1,)]

    @sy.func2plan(args_shape=args_shape, state=(th.tensor([1.0]),))
    def foo(x, state):
        (bias,) = state.read()
        x = x * 2
        return x + bias

    assert isinstance(foo.__str__(), str)
    assert len(foo.actions) > 0
    assert foo.is_built

    t = th.tensor([1.0, 2])
    x = foo(t)

    assert (x == th.tensor([3.0, 5])).all()

    args = PlaceHolder.create_placeholders(args_shape)
    torchscript_plan = th.jit.trace(foo, args)

    y = torchscript_plan(t)

    assert (y == th.tensor([3.0, 5])).all()


def test_func_plan_can_be_translated_to_torchscript(hook, workers):
    # Disable build time auto translation
    Plan._build_translators = []

    @sy.func2plan(args_shape=[(3, 3)])
    def plan(x):
        x = x * 2
        x = x.abs()
        return x

    orig_plan = plan.copy()

    inp = th.tensor([1, -1, 2])
    res1 = plan(inp)
    plan.add_translation(PlanTranslatorTorchscript)
    res2 = plan.torchscript(inp)
    assert (res1 == res2).all()

    # check that translation can be done after serde
    serde_plan = deserialize(serialize(orig_plan))
    serde_plan.add_translation(PlanTranslatorTorchscript)
    res3 = serde_plan.torchscript(inp)
    assert (res1 == res3).all()

    # check that translation is not lost after serde
    serde_plan_full = deserialize(serialize(plan))
    res4 = serde_plan_full.torchscript(inp)
    assert (res1 == res4).all()


def test_cls_plan_can_be_translated_to_torchscript(hook, workers):
    # Disable build time auto translation
    Plan._build_translators = []

    class Net(sy.Plan):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(2, 3)
            self.fc2 = nn.Linear(3, 1)

        def forward(self, x):
            x = self.fc1(x)
            x = F.relu(x)
            x = self.fc2(x)
            return x

    plan = Net()
    plan.build(th.zeros(10, 2))
    orig_plan = plan.copy()

    inp = th.randn(10, 2)

    res1 = plan(inp)
    plan.add_translation(PlanTranslatorTorchscript)
    res2 = plan.torchscript(inp, plan.parameters())
    assert (res1 == res2).all()

    # check that translation can be done after serde
    serde_plan = deserialize(serialize(orig_plan))
    serde_plan.add_translation(PlanTranslatorTorchscript)
    res3 = serde_plan.torchscript(inp, serde_plan.parameters())
    assert (res1 == res3).all()

    # check that translation is not lost after serde
    serde_plan_full = deserialize(serialize(plan))
    res4 = serde_plan_full.torchscript(inp, serde_plan_full.parameters())
    assert (res1 == res4).all()


def test_plan_translation_remove(hook, workers):
    # Disable build time auto translation
    Plan._build_translators = []

    @sy.func2plan(args_shape=[(3, 3)])
    def plan(x):
        x = x * 2
        x = x.abs()
        return x

    plan.add_translation(PlanTranslatorTorchscript)

    full_plan = plan.copy()
    assert full_plan.torchscript is not None

    assert plan.torchscript is not None
    assert len(plan.role.actions) > 0

    plan.remove_translation()
    assert plan.torchscript is not None
    assert len(plan.role.actions) == 0

    plan.remove_translation(PlanTranslatorTorchscript)
    assert plan.torchscript is None
    assert len(plan.role.actions) == 0

    full_plan.remove_translation(PlanTranslatorTorchscript)
    assert full_plan.torchscript is None
    assert len(full_plan.role.actions) > 0


def test_plan_translated_on_build(hook, workers):
    # Enable torchscript translator
    Plan.register_build_translator(PlanTranslatorTorchscript)

    @sy.func2plan(args_shape=[(3, 3)])
    def plan(x):
        x = x * 2
        x = x.abs()
        return x

    inp = th.tensor([1, -1, 2])
    res1 = plan(inp)
    res2 = plan.torchscript(inp)
    assert (res1 == res2).all()


def test_backward_autograd_can_be_translated(hook, workers):
    @sy.func2plan(args_shape=[(5, 5)], trace_autograd=True)
    def autograd_test(X):
        y = X * 5
        y = -y.log() / 2
        y = y.sum()
        y.backward()
        return X.grad

    X = th.ones(5, 5, requires_grad=True)

    # Result of torch autograd
    torch_grads = autograd_test(X)

    # Translate to torchscript
    autograd_test.add_translation(PlanTranslatorTorchscript)

    # Result of torchscript'ed traced backprop
    ts_plan_grads = autograd_test.torchscript(X)

    # (debug out)
    print("Torchscript Plan:\n", autograd_test.torchscript.code)

    # Test all results are equal
    assert torch_grads.eq(ts_plan_grads).all()
