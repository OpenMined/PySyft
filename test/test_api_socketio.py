import unittest
import time
import pytest
from random import randint, sample

import grid as gr
import syft as sy
import torch as th
import torch.nn.functional as F
import numpy as np

from test import PORTS, IDS, worker_ports
from test import conftest


hook = sy.TorchHook(th)


def test_host_plan_model(connected_node):
    class Net(sy.Plan):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = th.nn.Linear(2, 1)
            self.bias = th.tensor([1000.0])
            self.state += ["fc1", "bias"]

        def forward(self, x):
            x = self.fc1(x)
            return F.log_softmax(x, dim=0) + self.bias

    model = Net()
    model.build(th.tensor([1.0, 2]))

    nodes = list(connected_node.values())
    bob = nodes[0]
    bob.serve_model(model, model_id="1")

    # Call one time
    prediction = bob.run_inference(model_id="1", data=th.tensor([1.0, 2]))["prediction"]
    assert th.tensor(prediction) == th.tensor([1000.0])

    # Call one more time
    prediction = bob.run_inference(model_id="1", data=th.tensor([1.0, 2]))["prediction"]
    assert th.tensor(prediction) == th.tensor([1000.0])


def test_host_models_with_the_same_key(connected_node):
    class Net(sy.Plan):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = th.nn.Linear(2, 1)
            self.bias = th.tensor([1000.0])
            self.state += ["fc1", "bias"]

        def forward(self, x):
            x = self.fc1(x)
            return F.log_softmax(x, dim=1) + self.bias

    model = Net()
    model.build(th.tensor([1.0, 2]))

    nodes = list(connected_node.values())
    bob = nodes[0]

    # Serve model
    assert bob.serve_model(model, model_id="2")["success"]

    # Error when using the same id twice
    assert not bob.serve_model(model, model_id="2")["success"]


@pytest.mark.skipif(
    th.__version__ >= "1.1",
    reason="bug in pytorch version 1.1.0, jit.trace returns raw C function",
)
def test_host_jit_model(connected_node):
    class Net(th.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = th.nn.Linear(2, 1)
            self.bias = th.tensor([1000.0])

        def forward(self, x):
            x = self.fc1(x)
            return F.log_softmax(x, dim=1) + self.bias

    model = Net()
    trace_model = th.jit.trace(model, th.tensor([[1.0, 2]]))

    nodes = list(connected_node.values())
    bob = nodes[0]
    bob.serve_model(trace_model, model_id="1")

    # Call one time
    prediction = bob.run_inference(model_id="1", data=th.tensor([1.0, 2]))["prediction"]
    assert th.tensor(prediction) == th.tensor([1000.0])

    # Call one more time
    prediction = bob.run_inference(model_id="1", data=th.tensor([1.0, 2]))["prediction"]
    assert th.tensor(prediction) == th.tensor([1000.0])


def test_delete_model(connected_node):
    class Net(sy.Plan):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = th.nn.Linear(2, 1)
            self.bias = th.tensor([1000.0])
            self.state += ["fc1", "bias"]

        def forward(self, x):
            x = self.fc1(x)
            return F.log_softmax(x, dim=1) + self.bias

    model = Net()
    model.build(th.tensor([1.0, 2]))

    nodes = list(connected_node.values())
    bob = nodes[0]

    # Serve model
    assert bob.serve_model(model, model_id="test_delete_model")["success"]

    # Delete model
    assert bob.delete_model("test_delete_model")["success"]
    assert "test_delete_model" not in bob.models["models"]

    # Error when deleting again
    assert not bob.delete_model("test_delete_model")["success"]


def test_run_encrypted_model(connected_node):
    sy.hook.local_worker.is_client_worker = False

    class Net(sy.Plan):
        def __init__(self):
            super(Net, self).__init__()
            self.bias = th.tensor([2.0])
            self.state += ["bias"]

        def forward(self, x):
            # TODO: we're using a model that does not require
            # communication between nodes to compute.
            # Tests are breaking when communication
            # between nodes is required.
            return x + self.bias

    plan = Net()
    plan.build(th.tensor([1.0]))

    nodes = list(connected_node.values())

    bob, alice, james = nodes[:3]

    # Send plan
    plan.fix_precision().share(bob, james, crypto_provider=alice)

    # Share data
    x = th.tensor([1.0])
    x_sh = x.fix_precision().share(bob, james, crypto_provider=alice)

    # Execute the plan
    decrypted = plan(x_sh).get().float_prec()

    # Compare with local plan
    plan.get().float_precision()
    expected = plan(x)
    assert th.all(decrypted - expected.detach() < 1e-2)

    sy.hook.local_worker.is_client_worker = True


@pytest.mark.parametrize(
    "test_input, expected", [(node_id, node_id) for node_id in IDS]
)
def test_connect_nodes(test_input, expected, connected_node):
    assert connected_node[test_input].id == expected


def test_connect_node(connected_node):
    try:
        for node in connected_node:
            for n in connected_node:
                if n == node:
                    continue
                else:
                    connected_node[node].connect_grid_node(connected_node[n])
    except:
        unittest.TestCase.fail("test_connect_nodes : Exception raised!")


def test_connect_all_node(connected_node):
    workers = [connected_node[node] for node in connected_node]
    try:
        gr.connect_all_nodes(workers)
    except:
        unittest.TestCase.fail("test_connect_nodes : Exception raised!")


def test_send_tensor(connected_node):
    x = th.tensor([1.0, 0.4])
    x_s = x.send(connected_node["alice"])

    assert x_s.location.id == "alice"
    assert th.all(th.eq(x_s.get(), x))


def test_send_tag_tensor(connected_node):
    tag = "#tensor"
    description = "Tensor Description"
    x = th.tensor([[0.1, -4.3]]).tag(tag).describe(description)

    x_s = x.send(connected_node["alice"])
    x_s.child.garbage_collect_data = False

    assert x_s.description == description
    assert th.all(th.eq(x_s.get(), x))


def test_move_tensor(connected_node):
    alice, bob = connected_node["alice"], connected_node["bob"]
    x = th.tensor([[-0.1, -1]])

    x_s = x.send(alice)
    assert x_s.location.id == "alice"

    x_mv = x_s.move(bob)
    assert x_mv.location.id == "bob"
    assert th.all(th.eq(x_mv.get(), x))


@pytest.mark.parametrize(
    "x, y,",
    [
        (th.tensor(1.0), th.tensor(-0.1)),
        (
            th.tensor([[0.1, 1.2], [2.2, 3.1], [4.9, 5.2]]),
            th.tensor([[-3.2, 1.21], [-8.4, 34.9], [43.9, 50.2]]),
        ),
        (
            th.tensor(
                [
                    [0.9039, 0.6291, 1.0795],
                    [0.1586, 2.1939, -0.4900],
                    [-0.1909, -0.7503, 1.9355],
                ]
            ),
            th.zeros(3),
        ),
    ],
)
def test_add_remote_tensors(x, y, connected_node):
    result = x + y

    x_s = x.send(connected_node["alice"])
    y_s = y.send(connected_node["alice"])

    result_s = x_s + y_s
    assert result_s.get().tolist() == result.tolist()


@pytest.mark.parametrize(
    "x, y,",
    [
        (th.tensor(1.0), th.tensor(-0.1)),
        (
            th.tensor([[0.1, 1.2], [2.2, 3.1], [4.9, 5.2]]),
            th.tensor([[-3.2, 1.21], [-8.4, 34.9], [43.9, 50.2]]),
        ),
        (
            th.tensor(
                [
                    [0.9039, 0.6291, 1.0795],
                    [0.1586, 2.1939, -0.4900],
                    [-0.1909, -0.7503, 1.9355],
                ]
            ),
            th.zeros(3),
        ),
    ],
)
def test_sub_remote_tensors(x, y, connected_node):
    result = x - y

    x_s = x.send(connected_node["alice"])
    y_s = y.send(connected_node["alice"])

    result_s = x_s - y_s
    assert result_s.get().tolist() == result.tolist()


@pytest.mark.parametrize(
    "x, y,",
    [
        (th.tensor(1.0), th.tensor(-0.1)),
        (
            th.tensor([[0.1, 1.2], [2.2, 3.1], [4.9, 5.2]]),
            th.tensor([[-3.2, 1.21], [-8.4, 34.9], [43.9, 50.2]]),
        ),
        (
            th.tensor(
                [
                    [0.9039, 0.6291, 1.0795],
                    [0.1586, 2.1939, -0.4900],
                    [-0.1909, -0.7503, 1.9355],
                ]
            ),
            th.zeros(3),
        ),
    ],
)
def test_mul_remote_tensors(x, y, connected_node):
    result = x * y

    x_s = x.send(connected_node["alice"])
    y_s = y.send(connected_node["alice"])

    result_s = x_s * y_s
    assert result_s.get().tolist() == result.tolist()


@pytest.mark.parametrize(
    "x, y,",
    [
        (th.tensor(1.0), th.tensor(-0.1)),
        (
            th.tensor([[0.1, 1.2], [2.2, 3.1], [4.9, 5.2]]),
            th.tensor([[-3.2, 1.21], [-8.4, 34.9], [43.9, 50.2]]),
        ),
        (
            th.tensor(
                [
                    [0.9039, 0.6291, 1.0795],
                    [0.1586, 2.1939, -0.4900],
                    [-0.1909, -0.7503, 1.9355],
                ]
            ),
            th.zeros(3) + 1,
        ),
    ],
)
def test_div_remote_tensors(x, y, connected_node):
    result = x / y

    x_s = x.send(connected_node["bob"])
    y_s = y.send(connected_node["bob"])

    result_s = x_s / y_s
    assert result_s.get().tolist() == result.tolist()


@pytest.mark.parametrize(
    "x, y,",
    [
        (th.tensor(1.0), 1),
        (th.tensor([[0.1, 1.2], [2.2, 3.1], [4.9, 5.2]]), 2),
        (
            th.tensor(
                [
                    [0.9039, 0.6291, 1.0795],
                    [0.1586, 2.1939, -0.4900],
                    [-0.1909, -0.7503, 1.9355],
                ]
            ),
            3,
        ),
    ],
)
def test_exp_remote_tensor(x, y, connected_node):
    result = x ** y

    x_s = x.send(connected_node["bob"])
    result_s = x_s ** y
    assert result_s.get().tolist() == result.tolist()


def test_share_tensors(connected_node):
    x = th.tensor([1, 2, 3, 4, 5, 6])
    x_s = x.share(*connected_node.values())

    for node_id in IDS:
        assert node_id in x_s.child.child
    assert x_s.get().tolist() == x.tolist()


@pytest.mark.parametrize(
    "x, y,",
    [
        (th.tensor(1.0), th.tensor(0.1)),
        (
            th.tensor([[0.1, 1.2], [2.2, 3.1], [4.9, 5.2]]),
            th.tensor([[-3.2, 1.21], [-8.4, 34.9], [43.9, 50.2]]),
        ),
        (
            th.tensor(
                [
                    [0.9039, 0.6291, 1.0795],
                    [0.1586, 2.1939, -0.4900],
                    [-0.1909, -0.7503, 1.9355],
                ]
            ),
            th.zeros(3),
        ),
    ],
)
def test_add_shared_tensors(x, y, connected_node):
    result = x + y

    x_s = x.fix_prec().share(*connected_node.values())
    y_s = y.fix_prec().share(*connected_node.values())

    result_s = x_s + y_s
    assert th.allclose(result_s.get().float_prec(), result, atol=1e-3)


@pytest.mark.parametrize(
    "x, y,",
    [
        (th.tensor(1.0), th.tensor(0.1)),
        (
            th.tensor([[0.1, 1.2], [2.2, 3.1], [4.9, 5.2]]),
            th.tensor([[-3.2, 1.21], [-8.4, 34.9], [43.9, 50.2]]),
        ),
        (
            th.tensor(
                [
                    [0.9039, 0.6291, 1.0795],
                    [0.1586, 2.1939, -0.4900],
                    [-0.1909, -0.7503, 1.9355],
                ]
            ),
            th.zeros(3),
        ),
    ],
)
def test_sub_shared_tensors(x, y, connected_node):
    result = x - y

    x_s = x.fix_prec().share(*connected_node.values())
    y_s = y.fix_prec().share(*connected_node.values())

    result_s = x_s - y_s
    assert th.allclose(result_s.get().float_prec(), result, atol=1e-3)


@pytest.mark.parametrize(
    "x, y,",
    [
        (
            th.tensor(
                [
                    [0.9039, 0.6291, 1.0795],
                    [0.1586, 2.1939, -0.4900],
                    [-0.1909, -0.7503, 1.9355],
                ]
            ),
            th.tensor(
                [
                    [0.9039, 0.6291, 1.0795],
                    [0.1586, 2.1939, -0.4900],
                    [-0.1909, -0.7503, 1.9355],
                ]
            ).t(),
        )
    ],
)
def test_mul_shared_tensors(x, y, connected_node):
    result = x.matmul(y)

    nodes = list(connected_node.values())
    bob, alice, james = nodes[:3]
    x_s = x.fix_prec().share(bob, alice, crypto_provider=james)
    y_s = y.fix_prec().share(bob, alice, crypto_provider=james)
    result_s = x_s.matmul(y_s)

    assert th.allclose(result_s.get().float_prec(), result, atol=1e-2)
