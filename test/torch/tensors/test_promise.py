import pytest
import torch
import syft


def test__str__():
    a = syft.Promise.FloatTensor(shape=torch.Size((3, 3)))
    assert isinstance(a.__str__(), str)


@pytest.mark.parametrize("cmd", ["__add__", "__sub__", "__mul__"])
def test_operations_between_promises(hook, cmd):
    hook.local_worker.is_client_worker = False

    a = syft.Promise.FloatTensor(shape=torch.Size((2, 2)))
    b = syft.Promise.FloatTensor(shape=torch.Size((2, 2)))

    actual = getattr(a, cmd)(b)

    ta = torch.tensor([[1.0, 2], [3, 4]])
    tb = torch.tensor([[-8.0, -7], [6, 5]])
    a.keep(ta)
    b.keep(tb)

    expected = getattr(ta, cmd)(tb)

    assert (actual.value() == expected).all()

    hook.local_worker.is_client_worker = True


@pytest.mark.parametrize("cmd", ["__add__", "__sub__", "__mul__"])
def test_operations_with_concrete(hook, cmd):
    hook.local_worker.is_client_worker = False

    a = syft.Promise.FloatTensor(shape=torch.Size((2, 2)))
    b = torch.tensor([[-8.0, -7], [6, 5]]).wrap()  # TODO fix need to wrap

    actual = getattr(a, cmd)(b)

    ta = torch.tensor([[1.0, 2], [3, 4]]).wrap()
    a.keep(ta)

    expected = getattr(ta, cmd)(b)

    assert (actual.value() == expected).all()

    hook.local_worker.is_client_worker = True


def test_send(workers):
    bob = workers["bob"]

    a = syft.Promise.FloatTensor(shape=torch.Size((2, 2)))

    x = a.send(bob)

    x.keep(torch.ones((2, 2)))

    assert (x.value().get() == torch.ones((2, 2))).all()


@pytest.mark.parametrize("cmd", ["__add__", "__sub__", "__mul__"])
def test_remote_operations(workers, cmd):
    bob = workers["bob"]

    a = syft.Promise.FloatTensor(shape=torch.Size((3, 3)))
    b = syft.Promise.FloatTensor(shape=torch.Size((3, 3)))

    x = a.send(bob)
    y = b.send(bob)

    actual = getattr(x, cmd)(y)

    tx = torch.tensor([[1.0, 2], [3, 4]])
    ty = torch.tensor([[-8.0, -7], [6, 5]])
    x.keep(tx)
    y.keep(ty)

    expected = getattr(tx, cmd)(ty)

    assert (actual.value().get() == expected).all()


def test_bufferized_results(hook):
    hook.local_worker.is_client_worker = False

    a = syft.Promise.FloatTensor(shape=torch.Size((3, 3)))

    a.keep(torch.ones(3, 3))
    a.keep(2 * torch.ones(3, 3))
    a.keep(3 * torch.ones(3, 3))

    assert (a.value() == torch.ones(3, 3)).all()
    assert (a.value() == 2 * torch.ones(3, 3)).all()
    assert (a.value() == 3 * torch.ones(3, 3)).all()

    hook.local_worker.is_client_worker = True


def test_plan_waiting_promise(hook, workers):
    hook.local_worker.is_client_worker = False

    @syft.func2plan(args_shape=[(3, 3)])
    def plan_test(data):
        return 2 * data + 1

    # Hack otherwise plan not found on local worker...
    hook.local_worker.register_obj(plan_test)

    a = syft.Promise.FloatTensor(shape=torch.Size((3, 3)))

    res = plan_test(a)

    a.keep(torch.ones(3, 3))

    assert (res.value() == 3 * torch.ones(3, 3)).all()

    # With non promises
    @syft.func2plan(args_shape=[(3, 3), (3, 3)])
    def plan_test(prom, tens):
        return prom + tens

    # Hack otherwise plan not found on local worker...
    hook.local_worker.register_obj(plan_test)

    a = syft.Promise.FloatTensor(shape=torch.Size((3, 3)))
    b = 2 * torch.ones(3, 3).wrap()

    res = plan_test(a, b)

    a.keep(torch.ones(3, 3).wrap())

    assert (res.value().child == 3 * torch.ones(3, 3)).all()

    # With several arguments and remote
    bob = workers["bob"]

    @syft.func2plan(args_shape=[(3, 3), (3, 3)])
    def plan_test_remote(in_a, in_b):
        return in_a + in_b

    a = syft.Promise.FloatTensor(shape=torch.Size((3, 3)))
    b = syft.Promise.FloatTensor(shape=torch.Size((3, 3)))

    x = b.send(bob)
    y = a.send(bob)
    ptr_plan = plan_test_remote.send(bob)

    res_ptr = ptr_plan(x, y)

    x.keep(torch.ones(3, 3))
    x.keep(3 * torch.ones(3, 3))

    y.keep(2 * torch.ones(3, 3))
    y.keep(4 * torch.ones(3, 3))

    assert (res_ptr.value().get() == 3 * torch.ones(3, 3)).all()
    assert (res_ptr.value().get() == 7 * torch.ones(3, 3)).all()

    hook.local_worker.is_client_worker = True


def test_protocol_waiting_promise(hook, workers):
    hook.local_worker.is_client_worker = False

    alice = workers["alice"]
    bob = workers["bob"]

    @syft.func2plan(args_shape=[(1,)])
    def plan_alice1(in_a):
        return in_a + 1

    @syft.func2plan(args_shape=[(1,)])
    def plan_bob1(in_b):
        return in_b + 2

    @syft.func2plan(args_shape=[(1,)])
    def plan_bob2(in_b):
        return in_b + 3

    @syft.func2plan(args_shape=[(1,)])
    def plan_alice2(in_a):
        return in_a + 4

    protocol = syft.Protocol(
        [("alice", plan_alice1), ("bob", plan_bob1), ("bob", plan_bob2), ("alice", plan_alice2)]
    )
    protocol.deploy(alice, bob)

    x = syft.Promise.FloatTensor(shape=torch.Size((1,)))
    in_ptr, res_ptr = protocol(x)

    in_ptr.keep(torch.tensor([1.0]))

    assert res_ptr.value().get() == 11

    hook.local_worker.is_client_worker = True
