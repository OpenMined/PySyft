import pytest
import torch as th

import syft as sy


def test_create_roles_from_decorator(workers):
    alice, bob = workers["alice"], workers["bob"]

    roles_args_shape = {"alice": ((1,),), "bob": ((1,),)}

    @sy.func2protocol(args_shape=roles_args_shape)
    def protocol(roles):
        # fetch tensors from stores
        tensor1 = roles["alice"].fetch(th.tensor([1]))
        tensor2 = roles["bob"].fetch(th.tensor([1]))

        t1plus = tensor1 + 1
        t2plus = tensor2 + 1

        return t1plus, t2plus

    assert len(protocol.roles) == 2
    assert "alice" in protocol.roles
    assert "bob" in protocol.roles


def test_multi_role_tracing(workers):
    alice, bob = workers["alice"], workers["bob"]

    @sy.func2protocol(args_shape={"alice": ((1,),), "bob": ((1,),)})
    def protocol(roles):
        # fetch tensors from stores
        tensor1 = roles["alice"].fetch(th.tensor([1]))
        tensor2 = roles["bob"].fetch(th.tensor([1]))

        t1plus = tensor1 + 1
        t2plus = tensor2 + 1

        return t1plus, t2plus

    protocol.build()

    assert protocol.is_built

    assert len(protocol.roles) == 2
    assert len(protocol.roles["alice"].actions) == 1
    assert len(protocol.roles["bob"].actions) == 1


def test_multi_role_execution(workers):
    alice, bob = workers["alice"], workers["bob"]

    @sy.func2protocol(args_shape={"alice": ((1,), (1,)), "bob": ((1,),)})
    def protocol(roles):
        tensor1 = roles["alice"].fetch(th.tensor([1]))
        tensor2 = roles["bob"].fetch(th.tensor([2]))
        tensor3 = roles["alice"].fetch(th.tensor([3]))

        res1 = tensor2
        res2 = tensor1 + tensor3
        res3 = tensor2 * 3

        return res1, res2, res3

    protocol.build()
    protocol.forward = None

    dict_res = protocol()

    assert (dict_res["bob"][0] == th.tensor([2])).all()
    assert (dict_res["bob"][1] == th.tensor([6])).all()
    assert (dict_res["alice"][0] == th.tensor([4])).all()


def test_copy(workers):
    alice, bob = workers["alice"], workers["bob"]

    @sy.func2protocol(args_shape={"alice": ((1,),), "bob": ((1,),)})
    def protocol(roles):
        # fetch tensors from stores
        tensor1 = roles["alice"].fetch(th.tensor([1]))
        tensor2 = roles["bob"].fetch(th.tensor([1]))

        t1plus = tensor1 + 1
        t2plus = tensor2 + 1

        return t1plus, t2plus

    protocol.build()
    copy = protocol.copy()

    assert copy.name == protocol.name
    assert copy.roles.keys() == protocol.roles.keys()
    assert [
        len(copy_role.actions) == len(role.actions)
        for copy_role, role in zip(copy.roles.values(), protocol.roles.values())
    ]
    assert copy.is_built == protocol.is_built
