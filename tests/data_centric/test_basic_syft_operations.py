import time
import unittest
from random import randint, sample

import numpy as np
import pytest
import syft as sy
import torch as th
from torch.nn import functional as F

from tests import IDS, PORTS, conftest, worker_ports

hook = sy.TorchHook(th)


@pytest.mark.skip
def test_host_plan_not_allowed_to_run_ops(connected_node):
    hook.local_worker.is_client_worker = False

    class Net(sy.Plan):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = th.nn.Linear(2, 1)
            self.bias = th.tensor([1000.0])

        def forward(self, x):
            x = self.fc1(x)
            return F.log_softmax(x, dim=0) + self.bias

    model = Net()
    model.build(th.tensor([1.0, 2]))

    nodes = list(connected_node.values())
    bob = nodes[0]
    bob.serve_model(
        model,
        model_id="not_allowed",
        allow_remote_inference=False,
        allow_download=False,
    )

    with pytest.raises(RuntimeError):
        bob.run_remote_inference(model_id="not_allowed", data=th.tensor([1.0, 2]))

    hook.local_worker.is_client_worker = True


@pytest.mark.skip
def test_host_plan_model(connected_node):
    hook.local_worker.is_client_worker = False

    class Net(sy.Plan):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = th.nn.Linear(2, 1)
            self.bias = th.tensor([1000.0])

        def forward(self, x):
            x = self.fc1(x)
            return F.log_softmax(x, dim=0) + self.bias

    model = Net()
    model.build(th.tensor([1.0, 2]))

    nodes = list(connected_node.values())
    bob = nodes[0]
    bob.serve_model(model, model_id="1", allow_remote_inference=True)

    # Call one time
    prediction = bob.run_remote_inference(model_id="1", data=th.tensor([1.0, 2]))
    assert th.tensor(prediction) == th.tensor([1000.0])

    # Call one more time
    prediction = bob.run_remote_inference(model_id="1", data=th.tensor([1.0, 2]))
    assert th.tensor(prediction) == th.tensor([1000.0])
    hook.local_worker.is_client_worker = True


@pytest.mark.skip
def test_host_models_with_the_same_key(connected_node):
    hook.local_worker.is_client_worker = False

    class Net(sy.Plan):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = th.nn.Linear(2, 1)
            self.bias = th.tensor([1000.0])

        def forward(self, x):
            x = self.fc1(x)
            return F.log_softmax(x, dim=1) + self.bias

    model = Net()
    model.build(th.tensor([1.0, 2]))

    nodes = list(connected_node.values())
    bob = nodes[0]

    # Serve model
    assert bob.serve_model(model, model_id="2")

    # Error when using the same id twice
    with pytest.raises(RuntimeError):
        bob.serve_model(model, model_id="2")
    hook.local_worker.is_client_worker = True


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
    bob.serve_model(trace_model, model_id="1", allow_remote_inference=True)

    # Call one time
    prediction = bob.run_remote_inference(model_id="1", data=th.tensor([1.0, 2]))
    assert th.tensor(prediction) == th.tensor([1000.0])

    # Call one more time
    prediction = bob.run_remote_inference(model_id="1", data=th.tensor([1.0, 2]))
    assert th.tensor(prediction) == th.tensor([1000.0])


@pytest.mark.skip
def test_delete_model(connected_node):
    hook.local_worker.is_client_worker = False

    class Net(sy.Plan):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = th.nn.Linear(2, 1)
            self.bias = th.tensor([1000.0])

        def forward(self, x):
            x = self.fc1(x)
            return F.log_softmax(x, dim=1) + self.bias

    model = Net()
    model.build(th.tensor([1.0, 2]))

    nodes = list(connected_node.values())
    bob = nodes[0]

    # Serve model
    assert bob.serve_model(model, model_id="test_delete_model")

    # Delete model
    assert bob.delete_model("test_delete_model")
    assert "test_delete_model" not in bob.models

    hook.local_worker.is_client_worker = True


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
                    connected_node[node].connect_nodes(connected_node[n])
    except:
        unittest.TestCase.fail("test_connect_nodes : Exception raised!")


def test_send_tensor(connected_node):
    x = th.tensor([1.0, 0.4])
    x_s = x.send(connected_node["Alice"])

    assert x_s.location.id == "Alice"
    assert th.all(th.eq(x_s.get(), x))


def test_send_private_tensor(connected_node):
    x = th.tensor([1.0, 0.4])

    # Private Tensor
    _x = x.private_tensor(allowed_users=["user"])

    # Pointer to private tensor.
    p_x = _x.send(connected_node["Alice"], user="user")
    assert p_x.location.id == "Alice"


def test_get_private_tensor(connected_node):
    x = th.tensor([1.0, 0.4])

    # Private Tensor
    _x = x.private_tensor(allowed_users=["user"])

    # Pointer to private tensor.
    p_x = _x.send(connected_node["Alice"], user="user")
    with pytest.raises(sy.exceptions.GetNotPermittedError):
        p_x.get()


def test_send_tag_tensor(connected_node):
    tag = "#tensor"
    description = "Tensor Description"
    x = th.tensor([[0.1, -4.3]]).tag(tag).describe(description)

    x_s = x.send(connected_node["Alice"])
    x_s.child.garbage_collect_data = False

    assert x_s.description == description
    assert th.all(th.eq(x_s.get(), x))


def test_move_tensor(connected_node):
    alice, bob = connected_node["Alice"], connected_node["Bob"]
    x = th.tensor([[-0.1, -1]])

    x_s = x.send(alice)
    assert x_s.location.id == "Alice"

    x_mv = x_s.move(bob)
    assert x_mv.location.id == "Bob"
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

    x_s = x.send(connected_node["Alice"])
    y_s = y.send(connected_node["Alice"])

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

    x_s = x.send(connected_node["Alice"])
    y_s = y.send(connected_node["Alice"])

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

    x_s = x.send(connected_node["Alice"])
    y_s = y.send(connected_node["Alice"])

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

    x_s = x.send(connected_node["Bob"])
    y_s = y.send(connected_node["Bob"])

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

    x_s = x.send(connected_node["Bob"])
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


@pytest.mark.skipif(
    condition=sy.version.__version__ >= "0.2.6",
    reason="Reported bug, see https://github.com/OpenMined/PySyft/issues/3648 AND https://github.com/OpenMined/PySyft/issues/3865",
)
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
