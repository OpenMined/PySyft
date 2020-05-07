import pytest

import torch

import syft as sy
from syft.execution.role import Role
from syft.execution.placeholder import PlaceHolder
from syft.execution.computation import ComputationAction


def test_register_computation_action():
    role = Role()
    placeholder = PlaceHolder()
    target = torch.ones([1])

    action = ("method_name", target, (), {})

    role.register_action((action, placeholder), ComputationAction)

    assert len(role.actions) == 1

    registered = role.actions[0]

    assert isinstance(registered, ComputationAction)
    assert registered.name == "method_name"
    assert registered.target == target
    assert registered.args == ()
    assert registered.kwargs == {}
