import pytest
import torch as th

import syft as sy


def test_trace_communication_actions(workers):
    alice, bob = workers["alice"], workers["bob"]

    @sy.func2protocol(args_shape=[(1,)])
    def protocol(tensor):
        tensor.send(bob)

        return tensor

    assert protocol.is_built

    assert len(protocol.role.actions) == 1
