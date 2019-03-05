import random

import syft as sy

import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F


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
