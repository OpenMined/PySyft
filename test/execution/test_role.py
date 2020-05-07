import pytest

import syft as sy
from syft.execution.role import Role
from syft.execution.placeholder import PlaceHolder
from syft.execution.computation import ComputationAction


def test_register_computation_action():
    role = Role()
    placeholder = PlaceHolder()

    action = ("method_name", None, (), {})

    role.register_action((action, placeholder), ComputationAction)

    assert len(role.actions) == 1
    assert role.actions[0].name == "method_name"
    assert role.actions[0].target == None
    assert role.actions[0].args == ()
    assert role.actions[0].kwargs == {}
