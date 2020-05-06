import pytest
import torch as th

import syft as sy


def test_multi_role_tracing(workers):
    alice, bob = workers["alice"], workers["bob"]

    @sy.func2protocol(args_shape=((1,), (1,)))
    def protocol(tensor1, tensor2):
        t1plus = tensor1 + 1
        t2plus = tensor2 + 1

        return t1plus, t2plus

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


def test_multi_role_execution(workers):
    alice, bob = workers["alice"], workers["bob"]

    @sy.func2protocol(args_shape=((1,), (1,), (1,)))
    def protocol(tensor1, tensor2, tensor3):
        res1 = tensor2
        res2 = tensor1 + tensor3
        res3 = tensor2 * 3

        return res1, res2, res3

    alice_tensor1 = th.tensor([1]).send(alice)
    bob_tensor2 = th.tensor([2]).send(bob)
    alice_tensor3 = th.tensor([3]).send(alice)

    # TODO temporary trick to tell during the protocol building to whom belongs the tensors
    # The fact that we send before build is not even used
    alice_tensor1.owner = alice
    bob_tensor2.owner = bob
    alice_tensor3.owner = alice

    protocol.build(alice_tensor1, bob_tensor2, alice_tensor3)
    protocol.forward = None

    dict_res = protocol(alice_tensor1, bob_tensor2, alice_tensor3)

    assert (dict_res["bob"][0].get() == th.tensor([2])).all()
    assert (dict_res["bob"][1].get() == th.tensor([6])).all()
    assert (dict_res["alice"][0].get() == th.tensor([4])).all()


def test_copy(workers):
    alice, bob = workers["alice"], workers["bob"]

    @sy.func2protocol(args_shape=((1,), (1,)))
    def protocol(tensor1, tensor2):
        t1plus = tensor1 + 1
        t2plus = tensor2 + 1

        return t1plus, t2plus

    alice_tensor = th.tensor([1]).send(alice)
    bob_tensor = th.tensor([1]).send(bob)
    # TODO temporary trick to tell during the protocol building to whom belongs the tensors
    alice_tensor.owner = alice
    bob_tensor.owner = bob

    protocol.build(alice_tensor, bob_tensor)

    copy = protocol.copy()

    assert copy.name == protocol.name
    assert copy.roles.keys() == protocol.roles.keys()
    assert [
        len(copy_role.actions) == len(role.actions)
        for copy_role, role in zip(copy.roles.values(), protocol.roles.values())
    ]
    assert copy.is_built == protocol.is_built
