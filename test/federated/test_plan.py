import syft as sy

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from syft.serde import deserialize
from syft.serde import serialize
import time

import pytest
import unittest.mock as mock

from syft.workers import WebsocketClientWorker
from syft.workers import WebsocketServerWorker


def test_plan_built_locally(hook):
    # To run a plan locally the local worker can't be a client worker,
    # since it needs to register objects
    hook.local_worker.is_client_worker = False

    @sy.func2plan
    def plan_abs(data):
        return data.abs()

    x = th.tensor([-1, 2, 3])
    _ = plan_abs(x)

    assert isinstance(plan_abs.__str__(), str)
    assert len(plan_abs.readable_plan) > 0


def test_plan_execute_locally(hook):
    @sy.func2plan
    def plan_abs(data):
        return data.abs()

    x = th.tensor([-1, 2, 3])
    x_abs = plan_abs(x)
    assert (x_abs == th.tensor([1, 2, 3])).all()


def test_plan_built_remotely(workers):
    bob = workers["bob"]
    alice = workers["alice"]

    @sy.func2plan
    def plan_abs(data):
        return data.abs()

    plan_abs.send(bob)
    x_ptr = th.tensor([-1, 7, 3]).send(bob)
    p = plan_abs(x_ptr)
    x_abs = p.get()

    assert (x_abs == th.tensor([1, 7, 3])).all()

    # Test send / get plan
    plan_abs.get()
    plan_abs.send(alice)

    x_ptr = th.tensor([-1, 2, 3]).send(alice)
    p = plan_abs(x_ptr)
    x_abs = p.get()
    assert (x_abs == th.tensor([1, 2, 3])).all()


def test_plan_built_on_method(hook):
    """
    Test @sy.meth2plan and plan send / get / send
    """
    x11 = th.tensor([-1, 2.0]).tag("input_data")
    x12 = th.tensor([1, -2.0]).tag("input_data2")
    x21 = th.tensor([-1, 2.0]).tag("input_data")
    x22 = th.tensor([1, -2.0]).tag("input_data2")

    device_1 = sy.VirtualWorker(hook, id="device_1", data=(x11, x12))
    device_2 = sy.VirtualWorker(hook, id="device_2", data=(x21, x22))

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

    net.send(device_1)
    net.forward.send(device_1)
    pointer_to_data = device_1.search("input_data")[0]
    pointer_to_result = net(pointer_to_data)
    pointer_to_result.get()

    net.get()
    net.forward.get()

    net.send(device_2)
    net.forward.send(device_2)
    pointer_to_data = device_2.search("input_data")[0]
    pointer_to_result = net(pointer_to_data)
    pointer_to_result.get()


def test_multiple_workers(workers):
    bob = workers["bob"]
    alice = workers["alice"]

    @sy.func2plan
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


def test_fetch_plan_built_locally(hook):
    hook.local_worker.is_client_worker = False

    @sy.func2plan
    def plan_mult_3(data):
        return data * 3

    x = th.tensor([-1, 2, 3])
    device_3 = sy.VirtualWorker(hook, id="device_3", data=(x, plan_mult_3))

    # Fetch plan
    fetched_plan = device_3.fetch_plan(plan_mult_3.id)
    assert isinstance(fetched_plan, sy.Plan)

    # Build and execute plan locally
    y = th.tensor([-1, 2, 3])
    assert (fetched_plan(y) == th.tensor([-3, 6, 9])).all()


def test_fetch_plan_built_remotely(hook):
    hook.local_worker.is_client_worker = False
    device_4 = sy.VirtualWorker(hook, id="device_4")

    @sy.func2plan
    def plan_mult_3(data):
        return data * 3

    x_ptr = th.tensor([-1, 2, 3]).send(device_4)

    # When you "send" a plan we don't actually send the
    # plan to the worker we just update the plan's location
    sent_plan = plan_mult_3.send(device_4)

    # When you execute the plan, we then send the plan to the
    # worker and build it
    _ = sent_plan(x_ptr)

    # Fetch plan
    fetched_plan = device_4.fetch_plan(sent_plan.id)
    get_plan = sent_plan.get()

    # Build plan and execute it locally
    x = th.tensor([-1, 2, 3])
    assert (get_plan(x) == th.tensor([-3, 6, 9])).all()
    assert (fetched_plan(x) == th.tensor([-3, 6, 9])).all()


def test_plan_serde(hook):
    hook.local_worker.is_client_worker = False

    @sy.func2plan
    def my_plan(data):
        x = data * 2
        y = (x - 2) * 10
        return x + y

    # TODO: remove this line when issue #2062 is fixed
    # Force to build plan
    my_plan(th.tensor([1, 2, 3]))

    serialized_plan = serialize(my_plan)
    deserialized_plan = deserialize(serialized_plan)

    x = th.tensor([-1, 2, 3])
    assert (deserialized_plan(x) == th.tensor([-42, 24, 46])).all()


def test_plan_execute_remotely(hook, start_proc):
    """Test plan execution remotely."""
    hook.local_worker.is_client_worker = False

    @sy.func2plan
    def my_plan(data):
        x = data * 2
        y = (x - 2) * 10
        return x + y

    # TODO: remove this line when issue #2062 is fixed
    # Force to build plan
    x = th.tensor([-1, 2, 3])
    my_plan(x)

    kwargs = {"id": "test_plan_worker", "host": "localhost", "port": 8768, "hook": hook}
    server = start_proc(WebsocketServerWorker, kwargs)

    time.sleep(0.1)
    socket_pipe = WebsocketClientWorker(**kwargs)

    plan_ptr = my_plan.send(socket_pipe)
    x_ptr = x.send(socket_pipe)
    plan_res = plan_ptr(x_ptr).get()

    assert (plan_res == th.tensor([-42, 24, 46])).all()

    # delete remote object before websocket connection termination
    del x_ptr

    server.terminate()


def test_replace_worker_ids_two_strings(hook):
    plan = sy.Plan(id="0", owner=hook.local_worker, name="test_plan")
    _replace_message_ids_orig = sy.federated.Plan._replace_message_ids
    mock_fun = mock.Mock(return_value=[])
    sy.federated.Plan._replace_message_ids = mock_fun
    plan.replace_worker_ids("me", "you")
    args = {"change_id": -1, "obj": [], "to_id": -1}
    calls = [
        mock.call(from_worker="me", to_worker="you", **args),
        mock.call(from_worker=b"me", to_worker=b"you", **args),
    ]
    assert len(mock_fun.mock_calls) == 2
    mock_fun.assert_has_calls(calls, any_order=True)
    sy.federated.Plan._replace_message_ids = _replace_message_ids_orig


def test_replace_worker_ids_one_string_one_int(hook):
    plan = sy.Plan(id="0", owner=hook.local_worker, name="test_plan")
    _replace_message_ids_orig = sy.federated.Plan._replace_message_ids

    mock_fun = mock.Mock(return_value=[])
    sy.federated.Plan._replace_message_ids = mock_fun
    plan.replace_worker_ids(100, "you")

    args = {"change_id": -1, "obj": [], "to_id": -1}
    calls = [mock.call(from_worker=100, to_worker="you", **args)]
    assert len(mock_fun.mock_calls) == 1
    mock_fun.assert_has_calls(calls, any_order=True)

    mock_fun = mock.Mock(return_value=[])
    sy.federated.Plan._replace_message_ids = mock_fun
    plan.replace_worker_ids("me", 200)
    calls = [
        mock.call(from_worker="me", to_worker=200, **args),
        mock.call(from_worker=b"me", to_worker=200, **args),
    ]
    assert len(mock_fun.mock_calls) == 2
    mock_fun.assert_has_calls(calls, any_order=True)
    sy.federated.Plan._replace_message_ids = _replace_message_ids_orig


def test_replace_worker_ids_two_ints(hook):
    plan = sy.Plan(id="0", owner=hook.local_worker, name="test_plan")
    _replace_message_ids_orig = sy.federated.Plan._replace_message_ids
    mock_fun = mock.Mock(return_value=[])
    sy.federated.Plan._replace_message_ids = mock_fun
    plan.replace_worker_ids(300, 400)
    args = {"change_id": -1, "obj": [], "to_id": -1}
    calls = [mock.call(from_worker=300, to_worker=400, **args)]
    mock_fun.assert_called_once()
    mock_fun.assert_has_calls(calls, any_order=True)
    sy.federated.Plan._replace_message_ids = _replace_message_ids_orig


def test__replace_message_ids():
    messages = [10, ("worker", "me"), "you", 20, 10, b"you", (30, ["you", "me", "bla"])]

    replaced = sy.federated.plan.Plan._replace_message_ids(
        obj=messages, change_id=10, to_id=100, from_worker="me", to_worker="another"
    )

    # note that tuples are converted to lists
    expected = [100, ["worker", "another"], "you", 20, 100, b"you", [30, ["you", "another", "bla"]]]

    assert replaced == expected


def test___call__function(hook):
    plan = sy.Plan(id="0", owner=hook.local_worker, name="test_plan")

    with pytest.raises(AssertionError):
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


def test___call__method(hook):
    plan = sy.Plan(id="0", owner=hook.local_worker, name="test_plan")

    with pytest.raises(AssertionError):
        plan(kwarg1="hello", kwarg2=None)

    result_id = 444

    pop_function_original = sy.ID_PROVIDER.pop
    sy.ID_PROVIDER.pop = mock.Mock(return_value=result_id)

    return_value = "return value"
    plan.execute_plan = mock.Mock(return_value=return_value)

    # test the method case
    self_value = "my_self"
    plan.self = self_value

    arg_list = (100, 200, 356)
    ret_val = plan(*arg_list)

    expected_args = [self_value] + list(arg_list)
    plan.execute_plan.assert_called_with(expected_args, [result_id])

    assert ret_val == return_value

    # reset function
    sy.ID_PROVIDER.pop = pop_function_original
