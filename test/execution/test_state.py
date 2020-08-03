import pytest
import torch as th
import torch.nn as nn
import torch.nn.functional as F

import syft as sy


def test_stateful_plan_built_automatically(hook):
    @sy.func2plan(args_shape=[(1,)], state=(th.tensor([1.0]),))
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


def test_stateful_plan_build(hook):
    @sy.func2plan(state=(th.tensor([1.0]),))
    def foo(x, state):
        (bias,) = state.read()
        x = x * 2
        return x + bias

    t = th.tensor([1.0, 2])
    x = foo(t)

    assert (x == th.tensor([3.0, 5])).all()


def test_add_to_state():
    class Net(sy.Plan):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(2, 3)
            self.fc2 = th.tensor([1.0])

        def forward(self, x):
            pass  # pragma: no cover

    model = Net()

    assert len(model.state.state_placeholders) == 3


def test_stateful_plan_method_execute_locally(hook):
    class Net(sy.Plan):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(2, 1)
            self.bias = th.tensor([1000.0])

        def forward(self, x):
            x = self.fc1(x)
            return F.log_softmax(x, dim=0) + self.bias

    model = Net()

    model.build(th.tensor([1.0, 2]))

    # Call one time
    assert model(th.tensor([1.0, 2])) == th.tensor([1000.0])

    # Call one more time
    assert model(th.tensor([1.0, 2.1])) == th.tensor([1000.0])


def test_stateful_plan_multiple_send(hook, workers):
    bob, alice = workers["bob"], workers["alice"]

    @sy.func2plan(args_shape=[(1,)], state=(th.tensor([1.0]),))
    def plan_abs(x, state):
        (bias,) = state.read()
        x = x.abs()
        return x + bias

    plan_ptr = plan_abs.send(bob)
    x_ptr = th.tensor([-1.0, 7, 3]).send(bob)
    p = plan_ptr(x_ptr)
    res = p.get()

    assert (res == th.tensor([2.0, 8, 4])).all()

    # Test get / send plan
    plan_ptr = plan_abs.send(alice)

    x_ptr = th.tensor([-1.0, 2, 3]).send(alice)
    p = plan_ptr(x_ptr)
    res = p.get()
    assert (res == th.tensor([2.0, 3, 4])).all()


def test_stateful_plan_multiple_workers(hook, workers):
    bob, alice = workers["bob"], workers["alice"]

    @sy.func2plan(args_shape=[(1,)], state=(th.tensor([1]),))
    def plan_abs(x, state):
        (bias,) = state.read()
        x = x.abs()
        return x + bias

    plan_ptr = plan_abs.send(bob, alice)
    x_ptr = th.tensor([-1, 7, 3]).send(bob)
    p = plan_ptr(x_ptr)
    x_abs = p.get()
    assert (x_abs == th.tensor([2, 8, 4])).all()

    x_ptr = th.tensor([-1, 9, 3]).send(alice)
    p = plan_ptr(x_ptr)
    x_abs = p.get()
    assert (x_abs == th.tensor([2, 10, 4])).all()


@pytest.mark.parametrize("is_func2plan", [True, False])
def test_fetch_stateful_plan(hook, is_func2plan, workers):

    if is_func2plan:

        @sy.func2plan(args_shape=[(1,)], state=(th.tensor([1.0]),))
        def plan(data, state):
            (bias,) = state.read()
            return data * bias

    else:

        class Net(sy.Plan):
            def __init__(self):
                super(Net, self).__init__()
                self.fc1 = nn.Linear(1, 1)

            def forward(self, x):
                return self.fc1(x)

        plan = Net()
        plan.build(th.tensor([1.2]))

    alice = workers["alice"]
    plan_ptr = plan.send(alice)

    # Fetch plan
    fetched_plan = plan.owner.fetch_plan(plan_ptr.id_at_location, alice)

    # Execute it locally
    x = th.tensor([-1.26])
    assert th.all(th.eq(fetched_plan(x), plan(x)))
    # assert fetched_plan.state.state_placeholders != plan.state.state_placeholders #TODO

    # Make sure fetched_plan is using the actions
    assert fetched_plan.forward is None
    assert fetched_plan.is_built

    # Make sure plan is using the blueprint: forward
    assert plan.forward is not None


@pytest.mark.parametrize("is_func2plan", [True, False])
def test_fetch_stateful_plan_remote(hook, is_func2plan, start_remote_worker):

    server, remote_proxy = start_remote_worker(
        id=f"test_fetch_stateful_plan_remote_{is_func2plan}", hook=hook, port=8802
    )

    if is_func2plan:

        @sy.func2plan(args_shape=[(1,)], state=(th.tensor([3.0]),))
        def plan(data, state):
            (bias,) = state.read()
            return data * bias

    else:

        class Net(sy.Plan):
            def __init__(self):
                super(Net, self).__init__()
                self.fc1 = nn.Linear(1, 1)

            def forward(self, x):
                return self.fc1(x)

        plan = Net()
        plan.build(th.tensor([1.2]))

    x = th.tensor([-1.26])
    expected = plan(x)
    plan_ptr = plan.send(remote_proxy)

    # Fetch plan
    fetched_plan = plan.owner.fetch_plan(plan_ptr.id_at_location, remote_proxy)

    # Execute it locally
    assert th.all(th.eq(fetched_plan(x), expected))
    # assert fetched_plan.state.state_placeholders != plan.state.state_placeholders #TODO

    # Make sure fetched_plan is using the actions
    assert fetched_plan.forward is None
    assert fetched_plan.is_built

    # Make sure plan is using the blueprint: forward
    assert plan.forward is not None

    remote_proxy.close()
    server.terminate()


def test_binding_fix_precision_plan(hook):
    """
    Here we make sure the attributes of a plan are still bound to state
    elements when calling fix_precision
    """

    class Net(sy.Plan):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(1, 1)

        def forward(self, x):
            return self.fc1(x)

    plan = Net()
    plan.build(th.tensor([1.2]))
    original_weight = plan.fc1.weight.clone()

    plan.fix_precision()
    plan.fc1.weight.float_prec_()

    assert (plan.fc1.weight - original_weight) < 10e-2


def test_binding_encrypted_plan(hook, workers):
    """
    Here we make sure the attributes of a plan are still bound to state
    elements when calling fix_prec + share
    """

    alice, bob, charlie = (
        workers["alice"],
        workers["bob"],
        workers["charlie"],
    )

    class Net(sy.Plan):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(1, 1)

        def forward(self, x):
            return self.fc1(x)

    plan = Net()
    plan.build(th.tensor([1.2]))
    original_weight = plan.fc1.weight.clone()

    plan.fix_precision().share(alice, bob, crypto_provider=charlie)
    plan.fc1.weight.get_().float_prec_()

    assert (plan.fc1.weight - original_weight) < 10e-2


@pytest.mark.parametrize("is_func2plan", [True, False])
def test_fetch_encrypted_stateful_plan(hook, is_func2plan, workers):
    # TODO: this test is not working properly with remote workers.
    # We need to investigate why this might be the case.

    alice, bob, charlie, james = (
        workers["alice"],
        workers["bob"],
        workers["charlie"],
        workers["james"],
    )

    if is_func2plan:

        @sy.func2plan(args_shape=[(1,)], state=(th.tensor([3.0, 2.1]),))
        def plan(data, state):
            (bias,) = state.read()
            return data * bias

    else:

        class Net(sy.Plan):
            def __init__(self):
                super(Net, self).__init__()
                self.fc1 = nn.Linear(2, 1)

            def forward(self, x):
                return self.fc1(x)

        plan = Net()
        plan.build(th.rand(3, 2))

    x = th.rand(3, 2)
    expected = plan(x)

    plan.fix_precision().share(alice, bob, crypto_provider=charlie)
    ptr_plan = plan.send(james)

    # Fetch plan
    fetched_plan = plan.owner.fetch_plan(ptr_plan.id_at_location, james)

    # Execute the fetch plan
    x_sh = x.fix_precision().share(alice, bob, crypto_provider=charlie)
    decrypted = fetched_plan(x_sh).get().float_prec()

    # Compare with local plan
    assert th.all(decrypted - expected.detach() < 1e-2)
    # assert fetched_plan.state.state_placeholders != plan.state.state_placeholders #TODO

    for fetched_tensor, tensor in zip(fetched_plan.state.tensors(), plan.state.tensors()):
        assert ((fetched_tensor == tensor).get().float_prec() == 1).all()

    # Make sure fetched_plan is using the actions
    assert fetched_plan.forward is None
    assert fetched_plan.is_built

    # Make sure plan is using the blueprint: forward
    assert plan.forward is not None
