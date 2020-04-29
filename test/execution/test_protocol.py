import pytest
import torch as th

import syft as sy


def test_multi_role_tracing(workers):
    alice, bob = workers["alice"], workers["bob"]

    @sy.func2protocol(args_shape=((1,), (1,)))
    def protocol(tensor1, tensor2):
        t1plus = tensor1 + 1
        t2plus = tensor2 + 1

        return (t1plus, t2plus)

    alice_tensor = th.tensor([1]).send(alice)
    bob_tensor = th.tensor([1]).send(bob)

    # TODO temporary trick to tell during the protocol building to whom belongs the tensors
    alice_tensor.owner = alice
    bob_tensor.owner = bob

    protocol.build(alice_tensor, bob_tensor)

    assert protocol.is_built

    assert len(protocol.roles) == 2
    assert len(protocol.roles["alice"].actions) == 1
    assert len(protocol.roles["bob"].actions) == 1
