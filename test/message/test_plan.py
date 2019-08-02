import syft as sy

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from syft.serde.serde import deserialize
from syft.serde.serde import serialize
from syft import messaging
import time

import pytest
import unittest.mock as mock

from syft.workers import WebsocketClientWorker
from syft.workers import WebsocketServerWorker


def test_plan_built_automatically(hook):
    # To run a plan locally the local worker can't be a client worker,
    # since it needs to register objects
    hook.local_worker.is_client_worker = False

    @sy.func2plan(args_shape=[(1,)])
    def plan_abs(data):
        return data.abs()

    assert isinstance(plan_abs.__str__(), str)
    assert len(plan_abs.readable_plan) > 0
    assert plan_abs.is_built

    hook.local_worker.is_client_worker = True


def test_plan_build(hook):
    hook.local_worker.is_client_worker = False

    @sy.func2plan(args_shape=())
    def plan_abs(data):
        return data.abs()

    assert not plan_abs.is_built
    assert not len(plan_abs.readable_plan)

    plan_abs.build(th.tensor([-1]))

    assert len(plan_abs.readable_plan)
    assert plan_abs.is_built

    hook.local_worker.is_client_worker = True


def test_plan_built_automatically_with_any_dimension(hook):
    hook.local_worker.is_client_worker = False

    @sy.func2plan(args_shape=[(-1, 1)])
    def plan_abs(data):
        return data.abs()

    assert isinstance(plan_abs.__str__(), str)
    assert len(plan_abs.readable_plan) > 0

    hook.local_worker.is_client_worker = True


def test_raise_exception_for_invalid_shape(hook):
    hook.local_worker.is_client_worker = False

    with pytest.raises(ValueError):

        @sy.func2plan(args_shape=[(1, -20)])
        def _(data):
            return data  # pragma: no cover

    hook.local_worker.is_client_worker = True


def test_raise_exception_when_sending_unbuilt_plan(workers):
    me = workers["me"]
    me.is_client_worker = False

    bob = workers["bob"]

    @sy.func2plan()
    def plan(data):
        return data  # pragma: no cover

    with pytest.raises(RuntimeError):
        plan.send(bob)

    me.is_client_worker = True


def test_plan_execute_locally(hook):
    hook.local_worker.is_client_worker = False

    @sy.func2plan(args_shape=[(1,)])
    def plan_abs(data):
        return data.abs()

    x = th.tensor([-1, 2, 3])
    x_abs = plan_abs(x)
    assert (x_abs == th.tensor([1, 2, 3])).all()

    hook.local_worker.is_client_worker = True


def test_plan_method_execute_locally(hook):
    hook.local_worker.is_client_worker = False

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(2, 3)
            self.fc2 = nn.Linear(3, 2)
            self.fc3 = nn.Linear(2, 1)

        @sy.method2plan
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            x = self.fc3(x)
            return F.log_softmax(x)

    model = Net()

    # Force build
    assert model(th.tensor([1.0, 2])) == 0

    # Test call multiple times
    assert model(th.tensor([1.0, 2.1])) == 0

    hook.local_worker.is_client_worker = True


def test_plan_multiple_send(workers):
    me = workers["me"]
    me.is_client_worker = False

    bob = workers["bob"]
    alice = workers["alice"]

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
    me.is_client_worker = True


def test_plan_built_on_method(hook):
    """
    Test @sy.meth2plan and plan send / get / send
    """
    hook.local_worker.is_client_worker = False

    x11 = th.tensor([-1, 2.0]).tag("input_data")
    x21 = th.tensor([-1, 2.0]).tag("input_data")

    device_1 = sy.VirtualWorker(hook, id="device_1", data=(x11,))
    device_2 = sy.VirtualWorker(hook, id="device_2", data=(x21,))

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(2, 3)
            self.fc2 = nn.Linear(3, 2)

        @sy.method2plan
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return F.log_softmax(x, dim=0)

    net = Net()

    # build
    net.forward.build(th.tensor([1, 2.0]))

    net.send(device_1)
    pointer_to_data = device_1.search("input_data")[0]
    pointer_to_result = net(pointer_to_data)

    assert isinstance(pointer_to_result.get(), th.Tensor)

    net.get()
    net.send(device_2)

    pointer_to_data = device_2.search("input_data")[0]
    pointer_to_result = net(pointer_to_data)

    assert isinstance(pointer_to_result.get(), th.Tensor)

    hook.local_worker.is_client_worker = True


def test_multiple_workers(workers):
    me = workers["me"]
    me.is_client_worker = False
    bob = workers["bob"]
    alice = workers["alice"]

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

    me.is_client_worker = True


def test_fetch_plan(hook):
    hook.local_worker.is_client_worker = False
    device_4 = sy.VirtualWorker(hook, id="device_4")

    @sy.func2plan(args_shape=[(1,)])
    def plan_mult_3(data):
        return data * 3

    sent_plan = plan_mult_3.send(device_4)

    # Fetch plan
    fetched_plan = device_4.fetch_plan(sent_plan.id)
    get_plan = sent_plan.get()

    # Execut it locally
    x = th.tensor([-1, 2, 3])
    assert (get_plan(x) == th.tensor([-3, 6, 9])).all()
    assert fetched_plan.is_built
    assert (fetched_plan(x) == th.tensor([-3, 6, 9])).all()

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


def test_execute_plan_remotely(hook, start_proc):
    """Test plan execution remotely."""
    hook.local_worker.is_client_worker = False

    @sy.func2plan(args_shape=[(1,)])
    def my_plan(data):
        x = data * 2
        y = (x - 2) * 10
        return x + y

    x = th.tensor([-1, 2, 3])
    local_res = my_plan(x)

    kwargs = {"id": "test_plan_worker", "host": "localhost", "port": 8799, "hook": hook}
    server = start_proc(WebsocketServerWorker, **kwargs)

    time.sleep(0.1)
    socket_pipe = WebsocketClientWorker(**kwargs)

    plan_ptr = my_plan.send(socket_pipe)
    x_ptr = x.send(socket_pipe)
    plan_res = plan_ptr(x_ptr).get()

    assert (plan_res == local_res).all()

    # delete remote object before websocket connection termination
    del x_ptr

    server.terminate()
    hook.local_worker.is_client_worker = True


def test_execute_plan_module_remotely(hook, start_proc):
    """Test plan execution remotely."""
    hook.local_worker.is_client_worker = False

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(2, 3)
            self.fc2 = nn.Linear(3, 2)

        @sy.method2plan
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return F.log_softmax(x, dim=0)

    net = Net()

    x = th.tensor([-1, 2.0])
    local_res = net(x)
    assert not net.forward.is_built

    net.forward.build(x)

    kwargs = {"id": "test_plan_worker_2", "host": "localhost", "port": 8799, "hook": hook}
    server = start_proc(WebsocketServerWorker, **kwargs)

    time.sleep(0.1)
    socket_pipe = WebsocketClientWorker(**kwargs)

    plan_ptr = net.send(socket_pipe)
    x_ptr = x.send(socket_pipe)
    remote_res = plan_ptr(x_ptr).get()

    assert (remote_res == local_res).all()

    # delete remote object before websocket connection termination
    del x_ptr

    server.terminate()
    hook.local_worker.is_client_worker = True


def test_train_plan_locally_and_then_send_it(hook, start_proc):
    """Test training a plan locally and then executing it remotely."""
    hook.local_worker.is_client_worker = False

    # Create toy model
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(2, 3)
            self.fc2 = nn.Linear(3, 2)

        @sy.method2plan
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
    net.forward.build(x)

    kwargs = {"id": "test_plan_worker_3", "host": "localhost", "port": 8800, "hook": hook}
    server = start_proc(WebsocketServerWorker, **kwargs)

    time.sleep(0.1)
    socket_pipe = WebsocketClientWorker(**kwargs)

    plan_ptr = net.send(socket_pipe)
    x_ptr = x.send(socket_pipe)
    remote_res = plan_ptr(x_ptr).get()

    assert (remote_res == local_res).all()

    # delete remote object before websocket connection termination
    del x_ptr

    server.terminate()
    hook.local_worker.is_client_worker = True


def test_replace_worker_ids_two_strings(hook):
    plan = sy.Plan(id="0", owner=hook.local_worker, name="test_plan")
    _replace_message_ids_orig = messaging.Plan._replace_message_ids
    mock_fun = mock.Mock(return_value=[])
    messaging.Plan._replace_message_ids = mock_fun
    plan.replace_worker_ids("me", "you")
    args = {"change_id": -1, "obj": [], "to_id": -1}
    calls = [
        mock.call(from_worker="me", to_worker="you", **args),
        mock.call(from_worker=b"me", to_worker=b"you", **args),
    ]
    assert len(mock_fun.mock_calls) == 2
    mock_fun.assert_has_calls(calls, any_order=True)
    messaging.Plan._replace_message_ids = _replace_message_ids_orig


def test_replace_worker_ids_one_string_one_int(hook):
    plan = sy.Plan(id="0", owner=hook.local_worker, name="test_plan")
    _replace_message_ids_orig = messaging.Plan._replace_message_ids

    mock_fun = mock.Mock(return_value=[])
    messaging.Plan._replace_message_ids = mock_fun
    plan.replace_worker_ids(100, "you")

    args = {"change_id": -1, "obj": [], "to_id": -1}
    calls = [mock.call(from_worker=100, to_worker="you", **args)]
    assert len(mock_fun.mock_calls) == 1
    mock_fun.assert_has_calls(calls, any_order=True)

    mock_fun = mock.Mock(return_value=[])
    messaging.Plan._replace_message_ids = mock_fun
    plan.replace_worker_ids("me", 200)
    calls = [
        mock.call(from_worker="me", to_worker=200, **args),
        mock.call(from_worker=b"me", to_worker=200, **args),
    ]
    assert len(mock_fun.mock_calls) == 2
    mock_fun.assert_has_calls(calls, any_order=True)
    messaging.Plan._replace_message_ids = _replace_message_ids_orig


def test_replace_worker_ids_two_ints(hook):
    plan = sy.Plan(id="0", owner=hook.local_worker, name="test_plan")
    _replace_message_ids_orig = messaging.Plan._replace_message_ids
    mock_fun = mock.Mock(return_value=[])
    messaging.Plan._replace_message_ids = mock_fun
    plan.replace_worker_ids(300, 400)
    args = {"change_id": -1, "obj": [], "to_id": -1}
    calls = [mock.call(from_worker=300, to_worker=400, **args)]
    mock_fun.assert_called_once()
    mock_fun.assert_has_calls(calls, any_order=True)
    messaging.Plan._replace_message_ids = _replace_message_ids_orig


def test__replace_message_ids():
    messages = [10, ("worker", "me"), "you", 20, 10, b"you", (30, ["you", "me", "bla"])]

    replaced = messaging.Plan._replace_message_ids(
        obj=messages, change_id=10, to_id=100, from_worker="me", to_worker="another"
    )

    # note that tuples are converted to lists
    expected = (100, ("worker", "another"), "you", 20, 100, b"you", (30, ("you", "another", "bla")))

    assert replaced == expected


def test___call__function(hook):
    plan = sy.Plan(id="0", owner=hook.local_worker, name="test_plan")
    with pytest.raises(ValueError):
        plan(kwarg1="hello", kwarg2=None)

    result_id = 444

    pop_function_original = sy.ID_PROVIDER.pop
    sy.ID_PROVIDER.pop = mock.Mock(return_value=result_id)

    return_value = "return value"
    plan.execute_plan = mock.Mock(return_value=return_value)

    arg_list = (100, 200, 356)
    ret_val = plan(*arg_list)

    plan.execute_plan.assert_called_with(arg_list, [result_id])

    assert ret_val == return_value

    # reset function
    sy.ID_PROVIDER.pop = pop_function_original


def test__call__raise(hook):
    plan = sy.Plan(id="0", owner=hook.local_worker, name="test_plan")
    with pytest.raises(ValueError):
        plan(kwarg1="hello", kwarg2=None)


def test__call__for_method(hook):
    plan = sy.Plan(id="0", owner=hook.local_worker, name="test_plan", is_method=True)
    result_id = 444

    pop_function_original = sy.ID_PROVIDER.pop
    sy.ID_PROVIDER.pop = mock.Mock(return_value=result_id)

    return_value = "return value"
    plan.execute_plan = mock.Mock(return_value=return_value)

    self_value = mock.Mock()
    self_value.send = mock.Mock()
    plan._self = self_value

    arg_list = (100, 200, 356)
    ret_val = plan(*arg_list)

    expected_args = tuple([self_value] + list(arg_list))
    plan.execute_plan.assert_called_with(expected_args, [result_id])

    assert ret_val == return_value

    # reset function
    sy.ID_PROVIDER.pop = pop_function_original
