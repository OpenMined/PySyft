import pytest
import torch as th

import syft as sy
from syft.generic.frameworks.types import FrameworkTensor
from syft.generic.pointers.pointer_protocol import PointerProtocol
from syft.generic.pointers.pointer_tensor import PointerTensor


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

    protocol = sy.Protocol([("worker1", inc1), ("worker2", inc2), ("worker3", inc3)])
    return protocol


def test_deploy(workers):
    """
    This test validates the following scenario:
    A creates a protocol
    A deploys it on workers D, E and F
    """
    alice, bob, charlie = workers["alice"], workers["bob"], workers["charlie"]

    protocol = _create_inc_protocol()

    workers = alice, bob, charlie

    protocol.deploy(*workers)

    assert protocol.workers_resolved

    protocol._assert_is_resolved()

    # Assert the plan were sent to a consistent worker
    assert all(plan_ptr.location.id == worker.id for worker, plan_ptr in protocol.plans)

    # Assert the order of the worker was preserved
    assert all(
        plan_ptr.location.id == worker.id for (_, plan_ptr), worker in zip(protocol.plans, workers)
    )


def test_deploy_with_resolver(workers):
    """
    Like test_deploy, but now two of the three plans should be given to the same
    worker
    """
    alice, bob, charlie = workers["alice"], workers["bob"], workers["charlie"]

    protocol = _create_inc_protocol()
    worker3_plan = protocol.plans[2][1]
    protocol.plans[2] = ("worker1", worker3_plan)

    workers = alice, bob

    protocol.deploy(*workers)

    assert protocol.workers_resolved

    # Assert the plan were sent to a consistent worker
    assert all(plan_ptr.location.id == worker.id for worker, plan_ptr in protocol.plans)

    # Now test the error case
    protocol = _create_inc_protocol()

    with pytest.raises(RuntimeError):
        protocol.deploy(alice, bob)


def test_synchronous_run(workers):
    """
    This test validates the following scenario:
    A creates a protocol
    A deploys it on workers D, E and F
    A runs the protocol
    """
    alice, bob, charlie = workers["alice"], workers["bob"], workers["charlie"]

    protocol = _create_inc_protocol()

    protocol.deploy(alice, bob, charlie)

    x = th.tensor([1.0])
    ptr = protocol.run(x)

    assert ptr.location == charlie

    assert (
        isinstance(ptr, FrameworkTensor) and ptr.is_wrapper and isinstance(ptr.child, PointerTensor)
    )

    assert ptr.get() == th.tensor([4.0])


def test_synchronous_remote_run(workers):
    """
    This test validates the following scenario:
    A creates a protocol
    A deploys it on workers D, E and F
    A sends the protocol to the cloud C
    A asks a remote run on C
    """
    alice, bob, charlie, james = (
        workers["alice"],
        workers["bob"],
        workers["charlie"],
        workers["james"],
    )

    protocol = _create_inc_protocol()

    protocol.deploy(alice, bob, charlie)

    protocol.send(james)

    x = th.tensor([1.0]).send(james)
    ptr = protocol.run(x)

    assert ptr.location == james
    assert isinstance(ptr, FrameworkTensor) and ptr.is_wrapper

    ptr = ptr.get()

    assert ptr.location == charlie
    assert (
        isinstance(ptr, FrameworkTensor) and ptr.is_wrapper and isinstance(ptr.child, PointerTensor)
    )

    res = ptr.get()

    assert res == th.tensor([4.0])

    # BONUS: Error case when data is not correctly located

    x = th.tensor([1.0])
    with pytest.raises(RuntimeError):
        protocol.run(x)


def test_search_and_deploy(workers):
    """
    This test validates the following scenario:
    A creates a protocol (which is not deployed)
    A sends it to the cloud C
    B search C for a protocol and get it back
    B deploys the protocol on workers D, E and F
    B runs the protocol
    """
    alice, bob, charlie, james, me = (
        workers["alice"],
        workers["bob"],
        workers["charlie"],
        workers["james"],
        workers["me"],
    )

    protocol = _create_inc_protocol()

    protocol.send(james)

    ptr_protocol = me.request_search([protocol.id], location=james)[0]

    assert isinstance(ptr_protocol, PointerProtocol)

    protocol_back = ptr_protocol.get()

    protocol_back.deploy(alice, bob, charlie)

    x = th.tensor([1.0])
    ptr = protocol_back.run(x)

    assert ptr.location == charlie
    assert (
        isinstance(ptr, FrameworkTensor) and ptr.is_wrapper and isinstance(ptr.child, PointerTensor)
    )

    res = ptr.get()

    assert res == th.tensor([4.0])

    # BONUS: Re-send to cloud and run remotely

    james.clear_objects()
    protocol = protocol_back

    protocol.send(james)
    ptr_protocol = me.request_search([protocol.id], location=james)[0]
    x = th.tensor([1.0]).send(james)
    ptr = ptr_protocol.run(x)
    res = ptr.get().get()
    assert res == th.tensor([4.0])
