import pytest

import torch

import syft as sy
from syft.execution.role import Role
from syft.execution.placeholder import PlaceHolder
from syft.execution.computation import ComputationAction
from syft.execution.communication import CommunicationAction


def test_register_computation_action():
    role = Role()
    placeholder = PlaceHolder()
    target = torch.ones([1])

    action = ("__add__", target, (), {})

    role.register_action((action, placeholder), ComputationAction)

    assert len(role.actions) == 1

    registered = role.actions[0]

    assert isinstance(registered, ComputationAction)
    assert registered.name == "__add__"
    assert registered.target == target
    assert registered.args == ()
    assert registered.kwargs == {}
    assert registered.return_ids == (placeholder.id,)


def test_register_communication_action():
    role = Role()
    placeholder = PlaceHolder()
    target = torch.ones([1])

    action = ("get", target, (), {})

    role.register_action((action, placeholder), CommunicationAction)

    assert len(role.actions) == 1

    registered = role.actions[0]

    assert isinstance(registered, CommunicationAction)
    assert registered.name == "get"
    assert registered.target == target
    assert registered.args == ()
    assert registered.kwargs == {}

    assert registered.return_ids == (placeholder.id,)
