import pytest
import torch as th

import syft as sy


def test_trace_communication_actions_send(workers):
    alice, bob = workers["alice"], workers["bob"]

    @sy.func2protocol(args_shape=[(1,)])
    def protocol(tensor):
        # breakpoint()
        tensor.send(bob)

        return tensor

    assert protocol.is_built
    print(18, 'test', protocol.role.actions[0].name)
    assert len(protocol.role.actions) == 1


def test_trace_communication_actions_get(workers):
    alice, bob = workers["alice"], workers["bob"]

    @sy.func2protocol(args_shape=[(1,)])
    def protocol(x):
        ptr = x.send(bob)
        print(28, ptr)
        res = ptr.get()
        return res

    assert protocol.is_built
    print(18, 'test', protocol.role.actions,[a.name for a in  protocol.role.actions])

    assert len(protocol.role.actions) == 2

    # hook.local_worker.is_client_worker = True