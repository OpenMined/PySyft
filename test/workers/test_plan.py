import random

import syft as sy

import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F


def test_plan_built_locally():
    @sy.func2plan()
    def plan_abs(data):
        return data.abs()

    x = th.tensor([-1, 2, 3])
    x_abs = plan_abs(x)

    assert len(plan_abs.readable_plan) > 0


def test_plan_built_remotely(workers):
    bob = workers["bob"]
    alice = workers["alice"]

    @sy.func2plan()
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


# def test_plan_remote_function(workers):
#     plan_worker = workers["plan_worker"]
#
#     x = torch.tensor([1, -1, 3, 4])
#     x_ptr = x.send(plan_worker)
#     res_ptr = F.relu(x_ptr)
#     x_back = res_ptr.get()
#     assert (x_back == torch.tensor([1, 0, 3, 4])).all()
#
#
# def test_plan_remote_method(workers):
#     plan_worker = workers["plan_worker"]
#
#     x = torch.tensor([1, -1, 3, 4])
#     x_ptr = x.send(plan_worker)
#     res_ptr = x_ptr.abs()
#     x_back = res_ptr.get()
#     assert (x_back == torch.tensor([1, 1, 3, 4])).all()
#
#
# def test_plan_remote_inplace_method(workers):
#     plan_worker = workers["plan_worker"]
#
#     x = torch.tensor([1, -1, 3, 4])
#     x_ptr = x.send(plan_worker)
#     x_ptr.abs_()
#     x_back = x_ptr.get()
#     assert (x_back == torch.tensor([1, 1, 3, 4])).all()
