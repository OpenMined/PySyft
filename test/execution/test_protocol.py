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


def test_trace_communication_actions_get(workers):
    alice, bob = workers["alice"], workers["bob"]

    @sy.func2protocol(args_shape=[(1,)])
    def protocol(tensor):
        ptr = tensor.send(bob)
        print(29, type(ptr))
        res = ptr.get()
        print(31, "test", res)
        return res

    traced_actions = protocol.roles["me"].actions

    assert protocol.is_built
    assert len(traced_actions) == 2
    print([action.name for action in traced_actions])
    assert "get" in [action.name for action in traced_actions]


def test_trace_communication_actions_send(workers):
    alice, bob = workers["alice"], workers["bob"]

    @sy.func2protocol(args_shape=[(1,)])
    def protocol(tensor):
        ptr = tensor.send(bob)
        print(45, ptr)
        res = ptr.send(alice)
        return res

    traced_actions = protocol.roles["me"].actions
    print([action.name for action in traced_actions])

    assert protocol.is_built
    assert len(traced_actions) == 2
    assert "send" in [action.name for action in traced_actions]


def test_trace_communication_actions_move(workers):
    alice, bob = workers["alice"], workers["bob"]

    @sy.func2protocol(args_shape=[(1,)])
    def protocol(tensor):
        ptr = tensor.send(bob)
        res = ptr.move(alice)
        return res

    traced_actions = protocol.roles["me"].actions

    assert protocol.is_built
    assert len(traced_actions) == 2
    assert "move" in [action.name for action in traced_actions]


# def test_trace_communication_actions_share(workers):
#     alice, bob, charlie = workers["alice"], workers["bob"], workers["charlie"]

#     @sy.func2protocol(args_shape=[(1,)])
#     def protocol(tensor):
#         ptr = tensor.send(bob)
#         ptr = ptr.fix_prec()
#         # print(79, ptr)
#         res = ptr.share(alice, bob)
#         return res

#     traced_actions = protocol.roles["me"].actions

#     assert protocol.is_built
#     print([action.name for action in traced_actions])
#     assert len(traced_actions) == 2
#     assert "share" in [action.name for action in traced_actions]


# def test_trace_communication_actions_share_(workers):
#     alice, bob = workers["alice"], workers["bob"]

#     @sy.func2protocol(args_shape=[(1,)])
#     def protocol(tensor):
#         ptr = tensor.send(bob)
#         res = ptr.share_(alice, bob)
#         return res

#     traced_actions = protocol.roles["me"].actions

#     assert protocol.is_built
#     assert len(traced_actions) == 2
#     assert "share_" in [action.name for action in traced_actions]


def test_trace_communication_actions_remote_send(workers):
    alice, bob = workers["alice"], workers["bob"]

    @sy.func2protocol(args_shape=[(1,)])
    def protocol(tensor):
        ptr = tensor.send(bob)
        res = ptr.remote_send(alice)
        return res

    traced_actions = protocol.roles["me"].actions

    assert protocol.is_built
    assert len(traced_actions) == 2
    assert "remote_send" in [action.name for action in traced_actions]


def test_trace_communication_actions_mid_get(workers):
    alice, bob = workers["alice"], workers["bob"]

    @sy.func2protocol(args_shape=[(1,)])
    def protocol(tensor):
        ptr = tensor.send(bob)
        res = ptr.mid_get()
        return res

    traced_actions = protocol.roles["me"].actions

    assert protocol.is_built
    assert len(traced_actions) == 2
    assert "mid_get" in [action.name for action in traced_actions]


# def test_trace_communication_actions_remote_get(workers):
#     alice, bob = workers["alice"], workers["bob"]

#     @sy.func2protocol(args_shape=[(1,)])
#     def protocol(tensor):
#         ptr = tensor.send(bob)
#         res = ptr.remote_get()
#         return res

#     traced_actions = protocol.roles["me"].actions

#     assert protocol.is_built
#     assert len(traced_actions) == 2
#     print([action.name for action in traced_actions])
#     assert "remote_get" in [action.name for action in traced_actions]
