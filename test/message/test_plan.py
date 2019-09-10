import unittest.mock as mock

import pytest
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import syft as sy
from syft.generic.pointers.pointer_tensor import PointerTensor
from syft.messaging.plan import Plan
from syft.serde.serde import deserialize
from syft.serde.serde import serialize


def test_plan_built_automatically():
    @sy.func2plan(args_shape=[(1,)])
    def plan_abs(data):
        return data.abs()

    assert isinstance(plan_abs.__str__(), str)
    assert len(plan_abs.readable_plan) > 0
    assert plan_abs.is_built


def test_stateful_plan_built_automatically():
    @sy.func2plan(args_shape=[(1,)], state={"bias": th.tensor([1.0])})
    def foo(x, state):
        bias = state.read("bias")
        x = x * 2
        return x + bias

    assert isinstance(foo.__str__(), str)
    assert len(foo.readable_plan) > 0
    assert foo.is_built

    t = th.tensor([1.0, 2])
    x = foo(t)

    assert (x == th.tensor([3.0, 5])).all()


def test_plan_build():
    @sy.func2plan(args_shape=())
    def plan_abs(data):
        return data.abs()

    assert not plan_abs.is_built
    assert not len(plan_abs.readable_plan)

    plan_abs.build(th.tensor([-1]))

    assert len(plan_abs.readable_plan)
    assert plan_abs.is_built


def test_stateful_plan_build():
    @sy.func2plan(state={"bias": th.tensor([1.0])})
    def foo(x, state):
        bias = state.read("bias")
        x = x * 2
        return x + bias

    t = th.tensor([1.0, 2])
    x = foo(t)

    assert (x == th.tensor([3.0, 5])).all()


def test_plan_built_automatically_with_any_dimension():
    @sy.func2plan(args_shape=[(-1, 1)])
    def plan_abs(data):
        return data.abs()

    assert isinstance(plan_abs.__str__(), str)
    assert len(plan_abs.readable_plan) > 0


def test_raise_exception_for_invalid_shape():

    with pytest.raises(ValueError):

        @sy.func2plan(args_shape=[(1, -20)])
        def _(data):
            return data  # pragma: no cover


def test_raise_exception_when_sending_unbuilt_plan(workers):
    me, bob = workers["me"], workers["bob"]

    @sy.func2plan()
    def plan(data):
        return data  # pragma: no cover

    with pytest.raises(RuntimeError):
        plan.send(bob)


def test_plan_execute_locally():
    @sy.func2plan(args_shape=[(1,)])
    def plan_abs(data):
        return data.abs()

    x = th.tensor([-1, 2, 3])
    x_abs = plan_abs(x)
    assert (x_abs == th.tensor([1, 2, 3])).all()


def test_add_to_state():
    class Net(sy.Plan):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(2, 3)
            self.fc2 = nn.Linear(3, 2)
            self.state += ("fc1", "fc2")

        def forward(self, x):
            pass  # pragma: no cover

    model = Net()
    assert "fc1" in model.state.keys
    assert "fc2" in model.state.keys

    class Net(sy.Plan):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(2, 3)
            self.fc2 = nn.Linear(3, 2)
            self.state += ["fc1", "fc2"]

        def forward(self, x):
            pass  # pragma: no cover

    model = Net()
    assert "fc1" in model.state.keys
    assert "fc2" in model.state.keys

    class Net(sy.Plan):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(2, 3)
            self.fc2 = nn.Linear(3, 2)
            self.state.append("fc1")
            self.state.append("fc2")

        def forward(self, x):
            pass  # pragma: no cover

    model = Net()
    assert "fc1" in model.state.keys
    assert "fc2" in model.state.keys

    class Net(sy.Plan):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(2, 3)
            self.fc2 = nn.Linear(3, 2)
            self.add_to_state("fc1", "fc2")

        def forward(self, x):
            pass  # pragma: no cover

    model = Net()
    assert "fc1" in model.state.keys
    assert "fc2" in model.state.keys

    class Net(sy.Plan):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(2, 3)
            self.fc2 = nn.Linear(3, 2)
            self.add_to_state(["fc1", "fc2"])

        def forward(self, x):
            pass  # pragma: no cover

    model = Net()
    assert "fc1" in model.state.keys
    assert "fc2" in model.state.keys

    class Net(sy.Plan):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(2, 3)
            self.fc2 = nn.Linear(3, 2)
            self.state += ["fc3"]

        def forward(self, x):
            pass  # pragma: no cover

    with pytest.raises(AttributeError):
        model = Net()

    class Net(sy.Plan):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(2, 3)
            self.state += [self.fc1]

        def forward(self, x):
            pass  # pragma: no cover

    with pytest.raises(ValueError):
        model = Net()

    class Net(sy.Plan):
        def __init__(self):
            super(Net, self).__init__()
            self.y = "hello"
            self.state += ["y"]

        def forward(self, x):
            pass  # pragma: no cover

    with pytest.raises(ValueError):
        model = Net()


def test_plan_method_execute_locally():
    class Net(sy.Plan):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(2, 3)
            self.fc2 = nn.Linear(3, 2)
            self.fc3 = nn.Linear(2, 1)

            self.state += ["fc1", "fc2", "fc3"]

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            x = self.fc3(x)
            return F.log_softmax(x, dim=0)

    model = Net()

    model.build(th.tensor([1.0, 2]))

    # Call one time
    assert model(th.tensor([1.0, 2])) == 0

    # Call one more time
    assert model(th.tensor([1.0, 2.1])) == 0


def test_stateful_plan_method_execute_locally():
    class Net(sy.Plan):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(2, 1)
            self.bias = th.tensor([1000.0])

            self.state += ["fc1", "bias"]

        def forward(self, x):
            x = self.fc1(x)
            return F.log_softmax(x, dim=0) + self.bias

    model = Net()

    model.build(th.tensor([1.0, 2]))

    # Call one time
    assert model(th.tensor([1.0, 2])) == th.tensor([1000.0])

    # Call one more time
    assert model(th.tensor([1.0, 2.1])) == th.tensor([1000.0])


def test_plan_multiple_send(workers):
    me, bob, alice = workers["me"], workers["bob"], workers["alice"]

    @sy.func2plan(args_shape=[(1,)])
    def plan_abs(data):
        return data.abs()

    plan_abs.send(bob)
    x_ptr = th.tensor([-1, 7, 3]).send(bob)
    p = plan_abs(x_ptr)
    x_abs = p.get()

    assert (x_abs == th.tensor([1, 7, 3])).all()

    # Test get / send plan
    plan_abs.get()
    plan_abs.send(alice)

    x_ptr = th.tensor([-1, 2, 3]).send(alice)
    p = plan_abs(x_ptr)
    x_abs = p.get()
    assert (x_abs == th.tensor([1, 2, 3])).all()


def test_stateful_plan_multiple_send(workers):
    me, bob, alice = workers["me"], workers["bob"], workers["alice"]

    @sy.func2plan(args_shape=[(1,)], state={"bias": th.tensor([1.0])})
    def plan_abs(x, state):
        bias = state.read("bias")
        x = x.abs()
        return x + bias

    plan_abs.send(bob)
    x_ptr = th.tensor([-1.0, 7, 3]).send(bob)
    p = plan_abs(x_ptr)
    res = p.get()

    assert (res == th.tensor([2.0, 8, 4])).all()

    # Test get / send plan
    plan_abs.get()
    plan_abs.send(alice)

    x_ptr = th.tensor([-1.0, 2, 3]).send(alice)
    p = plan_abs(x_ptr)
    res = p.get()
    assert (res == th.tensor([2.0, 3, 4])).all()


def test_plan_built_on_class(hook):
    """
    Test class Plans and plan send / get / send
    """

    x11 = th.tensor([-1, 2.0]).tag("input_data")
    x21 = th.tensor([-1, 2.0]).tag("input_data")

    device_1 = sy.VirtualWorker(hook, id="device_1", data=(x11,))
    device_2 = sy.VirtualWorker(hook, id="device_2", data=(x21,))

    class Net(sy.Plan):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(2, 3)
            self.fc2 = nn.Linear(3, 1)

            self.bias = th.tensor([1000.0])

            self.state += ["fc1", "fc2", "bias"]

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return F.log_softmax(x, dim=0) + self.bias

    net = Net()

    # build
    net.build(th.tensor([1, 2.0]))

    net.send(device_1)
    pointer_to_data = device_1.search("input_data")[0]
    pointer_to_result = net(pointer_to_data)

    result = pointer_to_result.get()
    assert isinstance(result, th.Tensor)
    assert result == th.tensor([1000.0])

    net.get()
    net.send(device_2)

    pointer_to_data = device_2.search("input_data")[0]
    pointer_to_result = net(pointer_to_data)

    result = pointer_to_result.get()
    assert isinstance(result, th.Tensor)
    assert result == th.tensor([1000.0])


def test_multiple_workers(workers):
    me, bob, alice = workers["me"], workers["bob"], workers["alice"]

    @sy.func2plan(args_shape=[(1,)])
    def plan_abs(data):
        return data.abs()

    plan_abs.send(bob, alice)
    x_ptr = th.tensor([-1, 7, 3]).send(bob)
    p = plan_abs(x_ptr)
    x_abs = p.get()
    assert (x_abs == th.tensor([1, 7, 3])).all()

    x_ptr = th.tensor([-1, 9, 3]).send(alice)
    p = plan_abs(x_ptr)
    x_abs = p.get()
    assert (x_abs == th.tensor([1, 9, 3])).all()


def test_stateful_plan_multiple_workers(workers):
    me, bob, alice = workers["me"], workers["bob"], workers["alice"]

    @sy.func2plan(args_shape=[(1,)], state={"bias": th.tensor([1])})
    def plan_abs(x, state):
        bias = state.read("bias")
        x = x.abs()
        return x + bias

    plan_abs.send(bob, alice)
    x_ptr = th.tensor([-1, 7, 3]).send(bob)
    p = plan_abs(x_ptr)
    x_abs = p.get()
    assert (x_abs == th.tensor([2, 8, 4])).all()

    x_ptr = th.tensor([-1, 9, 3]).send(alice)
    p = plan_abs(x_ptr)
    x_abs = p.get()
    assert (x_abs == th.tensor([2, 10, 4])).all()


def test_fetch_plan(hook, workers):
    alice = workers["alice"]

    hook.local_worker.is_client_worker = False

    @sy.func2plan(args_shape=[(1,)])
    def plan(data):
        return data * 3

    plan.send(alice)

    # Fetch plan
    fetched_plan = plan.owner.fetch_plan(plan.id, alice)

    # Execute it locally
    x = th.tensor([-1.0, 2, 3])
    assert (plan(x) == th.tensor([-3.0, 6, 9])).all()
    assert (fetched_plan(x) == th.tensor([-3.0, 6, 9])).all()
    assert fetched_plan.forward is None
    assert fetched_plan.is_built

    hook.local_worker.is_client_worker = True


@pytest.mark.parametrize("is_func2plan", [True, False])
def test_fetch_stateful_plan(hook, is_func2plan, workers):
    hook.local_worker.is_client_worker = False

    if is_func2plan:

        @sy.func2plan(args_shape=[(1,)], state={"bias": th.tensor([3.0])})
        def plan(data, state):
            bias = state.read("bias")
            return data * bias

    else:

        class Net(sy.Plan):
            def __init__(self):
                super(Net, self).__init__()
                self.fc1 = nn.Linear(1, 1)
                self.add_to_state(["fc1"])

            def forward(self, x):
                return self.fc1(x)

        plan = Net()
        plan.build(th.tensor([1.2]))

    alice = workers["alice"]
    plan.send(alice)

    # Fetch plan
    fetched_plan = plan.owner.fetch_plan(plan.id, alice)

    # Execute it locally
    x = th.tensor([-1.26])
    assert th.all(th.eq(fetched_plan(x), plan(x)))
    assert fetched_plan.state_ids != plan.state_ids

    # Make sure fetched_plan is using the readable_plan
    assert fetched_plan.forward is None
    assert fetched_plan.is_built

    # Make sure plan is using the blueprint: forward
    assert plan.forward is not None

    hook.local_worker.is_client_worker = True


@pytest.mark.parametrize("is_func2plan", [True, False])
def test_fetch_stateful_plan_remote(hook, is_func2plan, start_remote_worker):
    hook.local_worker.is_client_worker = False

    server, remote_proxy = start_remote_worker(
        id="test_fetch_stateful_plan_remote_{}".format(is_func2plan), hook=hook, port=8802
    )

    if is_func2plan:

        @sy.func2plan(args_shape=[(1,)], state={"bias": th.tensor([3.0])})
        def plan(data, state):
            bias = state.read("bias")
            return data * bias

    else:

        class Net(sy.Plan):
            def __init__(self):
                super(Net, self).__init__()
                self.fc1 = nn.Linear(1, 1)
                self.add_to_state(["fc1"])

            def forward(self, x):
                return self.fc1(x)

        plan = Net()
        plan.build(th.tensor([1.2]))

    plan.send(remote_proxy)

    # Fetch plan
    fetched_plan = plan.owner.fetch_plan(plan.id, remote_proxy)

    # Execute it locally
    x = th.tensor([-1.26])
    assert th.all(th.eq(fetched_plan(x), plan(x)))
    assert fetched_plan.state_ids != plan.state_ids

    # Make sure fetched_plan is using the readable_plan
    assert fetched_plan.forward is None
    assert fetched_plan.is_built

    # Make sure plan is using the blueprint: forward
    assert plan.forward is not None

    remote_proxy.close()
    server.terminate()

    hook.local_worker.is_client_worker = True


@pytest.mark.parametrize("is_func2plan", [True, False])
def test_fetch_encrypted_stateful_plan(hook, is_func2plan, workers):
    # TODO: this test is not working properly with remote workers.
    # We need to investigate why this might be the case.
    hook.local_worker.is_client_worker = False

    alice, bob, charlie, james = (
        workers["alice"],
        workers["bob"],
        workers["charlie"],
        workers["james"],
    )

    if is_func2plan:

        @sy.func2plan(args_shape=[(1,)], state={"bias": th.tensor([3.0])})
        def plan(data, state):
            bias = state.read("bias")
            return data * bias

    else:

        class Net(sy.Plan):
            def __init__(self):
                super(Net, self).__init__()
                self.fc1 = nn.Linear(1, 1)
                self.add_to_state(["fc1"])

            def forward(self, x):
                return self.fc1(x)

        plan = Net()
        plan.build(th.tensor([1.2]))

    plan.fix_precision().share(alice, bob, crypto_provider=charlie).send(james)

    # Fetch plan
    fetched_plan = plan.owner.fetch_plan(plan.id, james)

    # Execute the fetch plan
    x = th.tensor([-1.0])
    x_sh = x.fix_precision().share(alice, bob, crypto_provider=charlie)
    decrypted = fetched_plan(x_sh).get().float_prec()

    # Compare with local plan
    plan.get().get().float_precision()
    expected = plan(x)
    assert th.all(decrypted - expected.detach() < 1e-2)
    assert fetched_plan.state_ids != plan.state_ids

    # Make sure fetched_plan is using the readable_plan
    assert fetched_plan.forward is None
    assert fetched_plan.is_built

    # Make sure plan is using the blueprint: forward
    assert plan.forward is not None

    hook.local_worker.is_client_worker = True


@pytest.mark.parametrize("is_func2plan", [True, False])
def test_fecth_plan_multiple_times(hook, is_func2plan, workers):
    hook.local_worker.is_client_worker = False

    alice, bob, charlie, james = (
        workers["alice"],
        workers["bob"],
        workers["charlie"],
        workers["james"],
    )

    if is_func2plan:

        @sy.func2plan(args_shape=[(1,)], state={"bias": th.tensor([3.0])})
        def plan(data, state):
            bias = state.read("bias")
            return data * bias

    else:

        class Net(sy.Plan):
            def __init__(self):
                super(Net, self).__init__()
                self.fc1 = nn.Linear(1, 1)
                self.add_to_state(["fc1"])

            def forward(self, x):
                return self.fc1(x)

        plan = Net()
        plan.build(th.tensor([1.2]))

    plan.fix_precision().share(alice, bob, crypto_provider=charlie).send(james)

    # Fetch plan
    fetched_plan = plan.owner.fetch_plan(plan.id, james, copy=True)

    # Execute the fetch plan
    x = th.tensor([-1.0])
    x_sh = x.fix_precision().share(alice, bob, crypto_provider=charlie)
    decrypted1 = fetched_plan(x_sh).get().float_prec()

    # 2. Re-fetch Plan
    fetched_plan = plan.owner.fetch_plan(plan.id, james, copy=True)

    # Execute the fetch plan
    x = th.tensor([-1.0])
    x_sh = x.fix_precision().share(alice, bob, crypto_provider=charlie)
    decrypted2 = fetched_plan(x_sh).get().float_prec()

    assert th.all(decrypted1 - decrypted2 < 1e-2)

    hook.local_worker.is_client_worker = True


def test_fetch_plan_remote(hook, start_remote_worker):
    hook.local_worker.is_client_worker = False

    server, remote_proxy = start_remote_worker(id="test_fetch_plan_remote", hook=hook, port=8803)

    @sy.func2plan(args_shape=[(1,)], state={"bias": th.tensor([1.0])})
    def plan_mult_3(data, state):
        bias = state.read("bias")
        return data * 3 + bias

    plan_mult_3.send(remote_proxy)

    # Fetch plan
    fetched_plan = plan_mult_3.owner.fetch_plan(plan_mult_3.id, remote_proxy)

    # Execute it locally
    x = th.tensor([-1.0, 2, 3])
    assert (plan_mult_3(x) == th.tensor([-2.0, 7, 10])).all()
    assert (fetched_plan(x) == th.tensor([-2.0, 7, 10])).all()
    assert fetched_plan.forward is None
    assert fetched_plan.is_built

    remote_proxy.close()
    server.terminate()

    hook.local_worker.is_client_worker = True


def test_plan_serde(hook):
    hook.local_worker.is_client_worker = False

    @sy.func2plan(args_shape=[(1, 3)])
    def my_plan(data):
        x = data * 2
        y = (x - 2) * 10
        return x + y

    serialized_plan = serialize(my_plan)
    deserialized_plan = deserialize(serialized_plan)

    x = th.tensor([-1, 2, 3])
    assert (deserialized_plan(x) == th.tensor([-42, 24, 46])).all()

    hook.local_worker.is_client_worker = True


def test_execute_plan_remotely(hook, start_remote_worker):
    """Test plan execution remotely."""

    @sy.func2plan(args_shape=[(1,)])
    def my_plan(data):
        x = data * 2
        y = (x - 2) * 10
        return x + y

    x = th.tensor([-1, 2, 3])
    local_res = my_plan(x)

    server, remote_proxy = start_remote_worker(id="test_plan_worker", hook=hook, port=8799)

    plan_ptr = my_plan.send(remote_proxy)
    x_ptr = x.send(remote_proxy)
    plan_res = plan_ptr(x_ptr).get()

    assert (plan_res == local_res).all()

    # delete remote object before websocket connection termination
    del x_ptr

    remote_proxy.close()
    server.terminate()


def test_execute_plan_module_remotely(hook, start_remote_worker):
    """Test plan execution remotely."""

    class Net(sy.Plan):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(2, 3)
            self.fc2 = nn.Linear(3, 2)

            self.bias = th.tensor([1000.0])

            self.state += ["fc1", "fc2", "bias"]

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return F.log_softmax(x, dim=0) + self.bias

    net = Net()

    x = th.tensor([-1, 2.0])
    local_res = net(x)
    assert not net.is_built

    net.build(x)

    server, remote_proxy = start_remote_worker(id="test_plan_worker_2", port=8799, hook=hook)

    plan_ptr = net.send(remote_proxy)
    x_ptr = x.send(remote_proxy)
    remote_res = plan_ptr(x_ptr).get()

    assert (remote_res == local_res).all()

    # delete remote object before websocket connection termination
    del x_ptr

    remote_proxy.close()
    server.terminate()


def test_train_plan_locally_and_then_send_it(hook, start_remote_worker):
    """Test training a plan locally and then executing it remotely."""

    # Create toy model
    class Net(sy.Plan):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(2, 3)
            self.fc2 = nn.Linear(3, 2)

            self.state += ["fc1", "fc2"]

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return F.log_softmax(x, dim=0)

    net = Net()

    # Create toy data
    x = th.tensor([-1, 2.0])
    y = th.tensor([1.0])

    # Train Model
    opt = optim.SGD(params=net.parameters(), lr=0.01)
    previous_loss = None

    for _ in range(5):
        # 1) erase previous gradients (if they exist)
        opt.zero_grad()

        # 2) make a prediction
        pred = net(x)

        # 3) calculate how much we missed
        loss = ((pred - y) ** 2).sum()

        # 4) figure out which weights caused us to miss
        loss.backward()

        # 5) change those weights
        opt.step()

        if previous_loss is not None:
            assert loss < previous_loss

        previous_loss = loss

    local_res = net(x)
    net.build(x)

    server, remote_proxy = start_remote_worker(id="test_plan_worker_3", port=8800, hook=hook)

    plan_ptr = net.send(remote_proxy)
    x_ptr = x.send(remote_proxy)
    remote_res = plan_ptr(x_ptr).get()

    assert (remote_res == local_res).all()

    # delete remote object before websocket connection termination
    del x_ptr

    remote_proxy.close()
    server.terminate()


def test_replace_worker_ids_two_strings(hook):
    plan = sy.Plan(id="0", owner=hook.local_worker, name="test_plan")
    _replace_message_ids_orig = Plan._replace_message_ids
    mock_fun = mock.Mock(return_value=[])
    Plan._replace_message_ids = mock_fun
    plan.replace_worker_ids("me", "you")
    args = {"change_id": -1, "obj": [], "to_id": -1}
    calls = [
        mock.call(from_worker="me", to_worker="you", **args),
        mock.call(from_worker=b"me", to_worker=b"you", **args),
    ]
    assert len(mock_fun.mock_calls) == 2
    mock_fun.assert_has_calls(calls, any_order=True)
    Plan._replace_message_ids = _replace_message_ids_orig


def test_replace_worker_ids_one_string_one_int(hook):
    plan = sy.Plan(id="0", owner=hook.local_worker, name="test_plan")
    _replace_message_ids_orig = Plan._replace_message_ids

    mock_fun = mock.Mock(return_value=[])
    Plan._replace_message_ids = mock_fun
    plan.replace_worker_ids(100, "you")

    args = {"change_id": -1, "obj": [], "to_id": -1}
    calls = [mock.call(from_worker=100, to_worker="you", **args)]
    assert len(mock_fun.mock_calls) == 1
    mock_fun.assert_has_calls(calls, any_order=True)

    mock_fun = mock.Mock(return_value=[])
    Plan._replace_message_ids = mock_fun
    plan.replace_worker_ids("me", 200)
    calls = [
        mock.call(from_worker="me", to_worker=200, **args),
        mock.call(from_worker=b"me", to_worker=200, **args),
    ]
    assert len(mock_fun.mock_calls) == 2
    mock_fun.assert_has_calls(calls, any_order=True)
    Plan._replace_message_ids = _replace_message_ids_orig


def test_replace_worker_ids_two_ints(hook):
    plan = sy.Plan(id="0", owner=hook.local_worker, name="test_plan")
    _replace_message_ids_orig = Plan._replace_message_ids
    mock_fun = mock.Mock(return_value=[])
    Plan._replace_message_ids = mock_fun
    plan.replace_worker_ids(300, 400)
    args = {"change_id": -1, "obj": [], "to_id": -1}
    calls = [mock.call(from_worker=300, to_worker=400, **args)]
    mock_fun.assert_called_once()
    mock_fun.assert_has_calls(calls, any_order=True)
    Plan._replace_message_ids = _replace_message_ids_orig


def test__replace_message_ids():
    messages = [10, ("worker", "me"), "you", 20, 10, b"you", (30, ["you", "me", "bla"])]

    replaced = Plan._replace_message_ids(
        obj=messages, change_id=10, to_id=100, from_worker="me", to_worker="another"
    )

    # note that tuples are converted to lists
    expected = (100, ("worker", "another"), "you", 20, 100, b"you", (30, ("you", "another", "bla")))

    assert replaced == expected


def test_send_with_plan(workers):
    bob = workers["bob"]

    @sy.func2plan([th.Size((1, 3))])
    def plan_double_abs(x):
        x = x.send(bob)
        x = x + x
        x = th.abs(x)
        return x

    expected = th.tensor([4.0, 4.0, 4.0])
    ptr_result = plan_double_abs(th.tensor([-2.0, 2.0, 2.0]))
    assert isinstance(ptr_result.child, sy.PointerTensor)
    result = ptr_result.get()
    assert th.equal(result, expected)
