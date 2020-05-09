import pytest
import torch as th

import syft as sy


def test_trace_communication_actions(workers):
    alice, bob = workers["alice"], workers["bob"]

    @sy.func2protocol(args_shape=[(1,)])
    def protocol(tensor):
        tensor.send(bob)

        return tensor

    traced_actions = protocol.roles["me"].actions

    assert protocol.is_built
    assert len(traced_actions) == 1
    assert "send" in [action.name for action in traced_actions]
