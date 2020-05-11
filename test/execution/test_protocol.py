import pytest
import torch as th

import syft as sy


def test_trace_communication_actions(workers):
    bob = workers["bob"]

    @sy.func2protocol(args_shape={"alice": ((1,),)})
    def protocol(roles):
        tensor = roles["alice"].fetch(th.tensor([1]))

        tensor.send(bob)
        return tensor

    traced_actions = protocol.roles["alice"].actions

    assert protocol.is_built
    assert len(traced_actions) == 1
    assert "send" in [action.name for action in traced_actions]


def test_trace_communication_actions_get(workers):
    bob = workers["bob"]

    @sy.func2protocol(args_shape={"alice": ((1,),)})
    def protocol(roles):
        tensor = roles["alice"].fetch(th.tensor([1]))

        ptr = tensor.send(bob)
        res = ptr.get()
        return res

    traced_actions = protocol.roles["alice"].actions

    assert protocol.is_built
    assert len(traced_actions) == 2
    assert "get" in [action.name for action in traced_actions]


def test_trace_communication_actions_send(workers):
    alice, bob = workers["alice"], workers["bob"]

    @sy.func2protocol(args_shape={"alice": ((1,),)})
    def protocol(roles):
        tensor = roles["alice"].fetch(th.tensor([1]))

        ptr = tensor.send(bob)
        res = ptr.send(alice)
        return res

    traced_actions = protocol.roles["alice"].actions

    assert protocol.is_built
    assert len(traced_actions) == 2
    assert "send" in [action.name for action in traced_actions]


def test_trace_communication_actions_move(workers):
    alice, bob = workers["alice"], workers["bob"]

    @sy.func2protocol(args_shape={"alice": ((1,),)})
    def protocol(roles):
        tensor = roles["alice"].fetch(th.tensor([1]))

        ptr = tensor.send(bob)
        res = ptr.move(alice)
        return res

    traced_actions = protocol.roles["alice"].actions

    assert protocol.is_built
    assert len(traced_actions) == 2
    assert "move" in [action.name for action in traced_actions]


def test_trace_communication_actions_share(workers):
    alice, bob = workers["alice"], workers["bob"]

    @sy.func2protocol(args_shape={"alice": ((1,),)})
    def protocol(roles):
        tensor = roles["alice"].fetch(th.tensor([1]))

        ptr = tensor.send(bob)
        ptr = ptr.fix_prec()
        res = ptr.share(alice, bob)
        return res

    traced_actions = protocol.roles["alice"].actions

    assert protocol.is_built
    assert len(traced_actions) == 3
    assert "share" in [action.name for action in traced_actions]


def test_trace_communication_actions_share_(workers):
    alice, bob = workers["alice"], workers["bob"]

    @sy.func2protocol(args_shape={"alice": ((1,),)})
    def protocol(roles):
        tensor = roles["alice"].fetch(th.tensor([1]))

        ptr = tensor.send(bob)
        ptr = ptr.fix_prec()
        res = ptr.share_(alice, bob)
        return res

    traced_actions = protocol.roles["alice"].actions

    assert protocol.is_built
    assert len(traced_actions) == 3
    assert "share_" in [action.name for action in traced_actions]


def test_trace_communication_actions_remote_send(workers):
    alice, bob = workers["alice"], workers["bob"]

    @sy.func2protocol(args_shape={"alice": ((1,),)})
    def protocol(roles):
        tensor = roles["alice"].fetch(th.tensor([1]))

        ptr = tensor.send(bob)
        res = ptr.remote_send(alice)
        return res

    traced_actions = protocol.roles["alice"].actions

    assert protocol.is_built
    assert len(traced_actions) == 2
    assert "remote_send" in [action.name for action in traced_actions]


def test_trace_communication_actions_mid_get(workers):
    bob = workers["bob"]

    @sy.func2protocol(args_shape={"alice": ((1,),)})
    def protocol(roles):
        tensor = roles["alice"].fetch(th.tensor([1]))

        ptr = tensor.send(bob)
        res = ptr.mid_get()
        return res

    traced_actions = protocol.roles["alice"].actions

    assert protocol.is_built
    assert len(traced_actions) == 2
    assert "mid_get" in [action.name for action in traced_actions]


def test_trace_communication_actions_remote_get(workers):
    alice, bob = workers["alice"], workers["bob"]

    @sy.func2protocol(args_shape={"alice": ((1,),)})
    def protocol(roles):
        tensor = roles["alice"].fetch(th.tensor([1]))

        ptr = tensor.send(bob).send(alice)
        res = ptr.remote_get()
        return res

    traced_actions = protocol.roles["alice"].actions

    assert protocol.is_built
    assert len(traced_actions) == 3
    assert "remote_get" in [action.name for action in traced_actions]


def test_create_roles_from_decorator(workers):

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
