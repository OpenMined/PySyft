import torch as th
import syft as sy

from syft.generic.pointers.pointer_plan import PointerPlan
from syft.execution.plan import Plan


def test_create_pointer_to_plan(hook, workers):
    alice, bob, charlie = workers["alice"], workers["bob"], workers["charlie"]

    hook.local_worker.is_client_worker = False

    @sy.func2plan(args_shape=[(1,)], state=(th.tensor([1.0]),))
    def plan(x, state):
        (bias,) = state.read()
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
        (bias,) = state.read()
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
        (bias,) = state.read()
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


def test_pointer_plan_parameters(workers):
    bob, me = workers["bob"], workers["me"]

    me.is_client_worker = False

    class Net(sy.Plan):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = th.nn.Linear(2, 1)

        def forward(self, x):
            x = self.fc1(x)
            return x

    model = Net()
    model.build(th.tensor([[0.0, 0.0]]))
    model = model.send(bob)

    param_ptrs = model.parameters()

    assert len(param_ptrs) == 2

    for param_ptr in param_ptrs:
        assert param_ptr.is_wrapper
        assert isinstance(param_ptr.child, sy.PointerTensor)

    me.is_client_worker = True
