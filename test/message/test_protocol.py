import pytest
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import syft as sy
from syft.generic.pointers.pointer_tensor import PointerTensor
from syft.messaging.plan import Plan
from syft.serde.serde import deserialize
from syft.serde.serde import serialize


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
    alice, bob, charlie = workers["alice"], workers["bob"], workers["charlie"]

    protocol = _create_inc_protocol()

    workers = alice, bob, charlie

    protocol.deploy(*workers)

    assert protocol.workers_resolved

    protocol._assert_is_resolved()

    # Assert the plan were sent to a consistent worker
    assert all(plan.locations[0] == worker.id for worker, plan in protocol.plans)

    # Assert the order of the worker was preserved
    assert all(plan.locations[0] == worker.id for (_, plan), worker in zip(protocol.plans, workers))


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
    assert all(plan.locations[0] == worker.id for worker, plan in protocol.plans)

    # Now test the error case
    protocol = _create_inc_protocol()

    with pytest.raises(RuntimeError):
        protocol.deploy(alice, bob)


def test_synchronous_run(workers):
    alice, bob, charlie = workers["alice"], workers["bob"], workers["charlie"]

    protocol = _create_inc_protocol()

    protocol.deploy(alice, bob, charlie)

    x = th.tensor([1.0])
    r = protocol.run(x)

    assert r.location == charlie

    assert r.get() == th.tensor([4.0])


def test_synchronous_remote_run(workers):
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
    r = protocol.run(x)

    assert r.location == james

    r = r.get()

    assert r.location == charlie

    r = r.get()

    assert r == th.tensor([4.0])

    # Error case when data is not correctly located

    x = th.tensor([1.0])
    with pytest.raises(RuntimeError):
        protocol.run(x)
