import torch as th
import syft as sy

from syft.generic.pointers.pointer_protocol import PointerProtocol
from syft.messaging.protocol import Protocol


def _create_inc_protocol():
    @sy.func2plan(args_shape=[(1,)])
    def inc1(x):
        return x + 1

    @sy.func2plan(args_shape=[(1,)])
    def inc2(x):
        return x + 1

    @sy.func2plan(args_shape=[(1,)])
    def inc3(x):
        return x + 1

    protocol = Protocol([("worker1", inc1), ("worker2", inc2), ("worker3", inc3)])
    return protocol


def test_create_pointer_to_plan(workers):
    alice, bob, charlie, james = (
        workers["alice"],
        workers["bob"],
        workers["charlie"],
        workers["james"],
    )

    protocol = _create_inc_protocol()

    protocol.deploy(alice, bob, charlie)

    protocol.send(james)
    id_at_location = protocol.id

    protocol_ptr = PointerProtocol(location=james, id_at_location=id_at_location)

    x = th.tensor([1.0]).send(james)

    ptr = protocol_ptr.run(x)

    assert (ptr.get().get() == th.tensor([4.0])).all()


def test_search_protocol(workers):
    alice, bob, charlie, james, me = (
        workers["alice"],
        workers["bob"],
        workers["charlie"],
        workers["james"],
        workers["me"],
    )

    protocol = _create_inc_protocol()

    protocol.deploy(alice, bob, charlie)

    protocol.send(james)
    id_at_location = protocol.id

    protocol_ptr = me.request_search([id_at_location], location=james)[0]

    x = th.tensor([1.0]).send(james)

    ptr = protocol_ptr.run(x)

    assert (ptr.get().get() == th.tensor([4.0])).all()


def test_get_protocols(workers):
    alice, bob, charlie, james, me = (
        workers["alice"],
        workers["bob"],
        workers["charlie"],
        workers["james"],
        workers["me"],
    )
    me.is_client_worker = False

    protocol = _create_inc_protocol()

    protocol.deploy(alice, bob, charlie)

    protocol.send(james)
    id_at_location = protocol.id

    protocol_ptr = me.request_search([id_at_location], location=james)[0]

    protocol_back = protocol_ptr.get()

    assert isinstance(protocol_back, Protocol)

    x = th.tensor([1.0])
    ptr_charlie = protocol_back.run(x)

    assert (ptr_charlie.get() == th.tensor([4.0])).all()

    me.is_client_worker = True
