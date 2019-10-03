import torch as th
import syft as sy

from syft.generic.pointers.pointer_plan import PointerPlan
from syft.messaging.plan import Plan


def test_create_pointer_to_plan(hook, workers):
    alice, bob, charlie = workers["alice"], workers["bob"], workers["charlie"]

    hook.local_worker.is_client_worker = False

    @sy.func2plan(args_shape=[(1,)], state=(th.tensor([1.0]),))
    def plan(x, state):
        bias, = state.read()
        return x + bias

    plan.send(alice)
    id_at_location = plan.id

    plan_ptr = PointerPlan(location=alice, id_at_location=id_at_location)

    x = th.tensor([1.0]).send(alice)

    ptr = plan_ptr(x)

    assert (ptr.get() == th.tensor([2.0])).all()

    hook.local_worker.is_client_worker = True


def test_search_plan(hook, workers):

    alice, me = workers["alice"], workers["me"]
    me.is_client_worker = False

    @sy.func2plan(args_shape=[(1,)], state=(th.tensor([1.0]),))
    def plan(x, state):
        bias, = state.read()
        return x + bias

    plan.send(alice)
    id_at_location = plan.id

    plan_ptr = me.request_search([id_at_location], location=alice)[0]

    assert isinstance(plan_ptr, PointerPlan)

    x = th.tensor([1.0]).send(alice)
    ptr = plan_ptr(x)

    assert (ptr.get() == th.tensor([2.0])).all()

    me.is_client_worker = True


def test_get_plan(workers):
    alice, me = workers["alice"], workers["me"]
    me.is_client_worker = False

    @sy.func2plan(args_shape=[(1,)], state=(th.tensor([1.0]),))
    def plan(x, state):
        bias, = state.read()
        return x + bias

    plan.send(alice)
    id_at_location = plan.id
    plan_ptr = me.request_search([id_at_location], location=alice)[0]

    plan = plan_ptr.get()

    assert isinstance(plan, Plan)

    x = th.tensor([1.0])
    res = plan(x)

    assert (res == th.tensor([2.0])).all()

    me.is_client_worker = True
