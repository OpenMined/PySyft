import pytest
import torch as th

import syft as sy
from syft.execution.role import Role


def test_func2protocol_creates_roles():
    @sy.func2protocol(roles=["alice", "bob"])
    def protocol(alice, bob):
        tensor = alice.torch.tensor([1])

        return tensor

    assert protocol.is_built
    assert len(protocol.roles) == 2
    assert isinstance(protocol.roles["alice"], Role)
    assert isinstance(protocol.roles["bob"], Role)


def test_framework_methods_traced_by_role():
    @sy.func2protocol(roles=["alice", "bob"])
    def protocol(alice, bob):
        tensor1 = alice.torch.rand([4, 4])
        tensor2 = bob.torch.rand([4, 4])

        return tensor1, tensor2

    assert protocol.is_built

    for role in protocol.roles.values():
        assert len(role.actions) == 1
        assert "torch.rand" in [action.name for action in role.actions]


def test_trace_communication_actions_send():
    @sy.func2protocol(roles=["alice", "bob"], args_shape={"alice": ((1,),)})
    def protocol(alice, bob):
        tensor = alice.torch.tensor([1])

        tensor.send(bob.worker)
        return tensor

    traced_actions = protocol.roles["alice"].actions

    assert protocol.is_built
    assert len(traced_actions) == 2
    assert "send" in [action.name for action in traced_actions]


def test_trace_communication_actions_get():
    @sy.func2protocol(roles=["alice", "bob"], args_shape={"alice": ((1,),)})
    def protocol(alice, bob):
        tensor = alice.torch.tensor([1])

        ptr = tensor.send(bob.worker)
        res = ptr.get()
        return res

    traced_actions = protocol.roles["alice"].actions

    assert protocol.is_built
    assert len(traced_actions) == 3
    assert "get" in [action.name for action in traced_actions]


def test_trace_communication_actions_ptr_send():
    @sy.func2protocol(roles=["alice", "bob"], args_shape={"alice": ((1,),)})
    def protocol(alice, bob):
        tensor = alice.torch.tensor([1])

        ptr = tensor.send(bob.worker)
        res = ptr.send(alice.worker)
        return res

    traced_actions = protocol.roles["alice"].actions

    assert protocol.is_built
    assert len(traced_actions) == 3
    assert "send" in [action.name for action in traced_actions]


def test_trace_communication_actions_move():
    @sy.func2protocol(roles=["alice", "bob"], args_shape={"alice": ((1,),)})
    def protocol(alice, bob):
        tensor = alice.torch.tensor([1])

        ptr = tensor.send(bob.worker)
        res = ptr.move(alice.worker)
        return res

    traced_actions = protocol.roles["alice"].actions

    assert protocol.is_built
    assert len(traced_actions) == 3
    assert "move" in [action.name for action in traced_actions]


def test_trace_communication_actions_share():
    @sy.func2protocol(roles=["alice", "bob"], args_shape={"alice": ((1,),)})
    def protocol(alice, bob):
        tensor = alice.torch.tensor([1])

        ptr = tensor.send(bob.worker)
        ptr = ptr.fix_prec()
        res = ptr.share(alice.worker, bob.worker)
        return res

    traced_actions = protocol.roles["alice"].actions

    assert protocol.is_built
    assert len(traced_actions) == 4
    assert "share" in [action.name for action in traced_actions]


def test_trace_communication_actions_share_():
    @sy.func2protocol(roles=["alice", "bob"], args_shape={"alice": ((1,),)})
    def protocol(alice, bob):
        tensor = alice.torch.tensor([1])

        ptr = tensor.send(bob.worker)
        ptr = ptr.fix_prec()
        res = ptr.share_(alice.worker, bob.worker)
        return res

    traced_actions = protocol.roles["alice"].actions

    assert protocol.is_built
    assert len(traced_actions) == 4
    assert "share_" in [action.name for action in traced_actions]


def test_trace_communication_actions_remote_send():
    @sy.func2protocol(roles=["alice", "bob"], args_shape={"alice": ((1,),)})
    def protocol(alice, bob):
        tensor = alice.torch.tensor([1])

        ptr = tensor.send(bob.worker)
        res = ptr.remote_send(alice.worker)
        return res

    traced_actions = protocol.roles["alice"].actions

    assert protocol.is_built
    assert len(traced_actions) == 3
    assert "remote_send" in [action.name for action in traced_actions]


def test_trace_communication_actions_mid_get():
    @sy.func2protocol(roles=["alice", "bob"], args_shape={"alice": ((1,),)})
    def protocol(alice, bob):
        tensor = alice.torch.tensor([1])

        ptr = tensor.send(bob.worker)
        res = ptr.mid_get()
        return res

    traced_actions = protocol.roles["alice"].actions

    assert protocol.is_built
    assert len(traced_actions) == 3
    assert "mid_get" in [action.name for action in traced_actions]


def test_trace_communication_actions_remote_get():
    @sy.func2protocol(roles=["alice", "bob"], args_shape={"alice": ((1,),)})
    def protocol(alice, bob):
        tensor = alice.torch.tensor([1])

        ptr = tensor.send(bob.worker).send(alice.worker)
        res = ptr.remote_get()
        return res

    traced_actions = protocol.roles["alice"].actions

    assert protocol.is_built
    assert len(traced_actions) == 4
    assert "remote_get" in [action.name for action in traced_actions]


def test_create_roles_from_decorator():
    @sy.func2protocol(roles=["alice", "bob"], args_shape={"alice": ((1,),), "bob": ((1,),)})
    def protocol(alice, bob):
        tensor1 = alice.torch.tensor([1])
        tensor2 = bob.torch.tensor([2])

        t1plus = tensor1 + 1
        t2plus = tensor2 + 1

        return t1plus, t2plus

    assert len(protocol.roles) == 2
    assert "alice" in protocol.roles
    assert "bob" in protocol.roles


def test_multi_role_tracing():
    @sy.func2protocol(roles=["alice", "bob"], args_shape={"alice": ((1,),), "bob": ((1,),)})
    def protocol(alice, bob):
        tensor1 = alice.torch.tensor([1])
        tensor2 = bob.torch.tensor([2])

        t1plus = tensor1 + 1
        t2plus = tensor2 + 1

        return t1plus, t2plus

    protocol.build()

    assert protocol.is_built

    assert len(protocol.roles) == 2
    assert len(protocol.roles["alice"].actions) == 2
    assert len(protocol.roles["bob"].actions) == 2


def test_multi_role_execution():
    @sy.func2protocol(roles=["alice", "bob"], args_shape={"alice": ((1,), (1,)), "bob": ((1,),)})
    def protocol(alice, bob):
        tensor1 = alice.torch.tensor([1])
        tensor2 = bob.torch.tensor([2])
        tensor3 = alice.torch.tensor([3])

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


def test_stateful_protocol(workers):
    shapes = {"alice": ((1,),), "bob": ((1,),)}
    states = {"alice": (th.tensor([1]), th.tensor([3])), "bob": (th.tensor([5]),)}

    @sy.func2protocol(roles=["alice", "bob"], args_shape=shapes, states=states)
    def protocol(alice, bob):
        # fetch tensors from states
        tensor_a1, tensor_a2 = alice.load_state()
        (tensor_b1,) = bob.load_state()

        t1plus = tensor_a1 + tensor_a2
        t2plus = tensor_b1 + 1

        return t1plus, t2plus

    assert all(protocol.roles["alice"].state.state_placeholders[0].child == th.tensor([1]))
    assert all(protocol.roles["alice"].state.state_placeholders[1].child == th.tensor([3]))
    assert all(protocol.roles["bob"].state.state_placeholders[0].child == th.tensor([5]))


def test_copy():
    @sy.func2protocol(roles=["alice", "bob"], args_shape={"alice": ((1,),), "bob": ((1,),)})
    def protocol(alice, bob):
        tensor1 = alice.torch.tensor([1])
        tensor2 = bob.torch.tensor([2])

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
