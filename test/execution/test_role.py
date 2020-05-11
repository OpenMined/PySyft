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


def test_reset():
    role = Role()
    placeholder = PlaceHolder()
    target = torch.ones([1])

    action = ("get", target, (), {})

    role.register_action((action, placeholder), CommunicationAction)
    role.placeholders = {"ph_id1": PlaceHolder(), "ph_id2": PlaceHolder()}
    role.input_placeholder_ids = ("input1", "input2")
    role.output_placeholder_ids = ("output1",)

    assert len(role.actions) == 1
    assert len(role.placeholders) == 2
    assert role.input_placeholder_ids == ("input1", "input2")
    assert role.output_placeholder_ids == ("output1",)

    role.reset()

    assert len(role.actions) == 0
    assert len(role.placeholders) == 0
    assert role.input_placeholder_ids == ()
    assert role.output_placeholder_ids == ()
