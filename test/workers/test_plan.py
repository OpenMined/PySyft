import random

import syft as sy

import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F


def test_plan_built_locally():
    @sy.func2plan
    def plan_abs(data):
        return data.abs()

    x = th.tensor([-1, 2, 3])
    x_abs = plan_abs(x)

    assert isinstance(plan_abs.__str__(), str)
    assert len(plan_abs.readable_plan) > 0


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
    x11 = torch.tensor([-1, 2.0]).tag("input_data")
    x12 = torch.tensor([1, -2.0]).tag("input_data2")
    x21 = torch.tensor([-1, 2.0]).tag("input_data")
    x22 = torch.tensor([1, -2.0]).tag("input_data2")

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
