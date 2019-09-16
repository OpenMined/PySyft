import pytest
import torch
import syft


def test__str__():
    a = syft.Promises.FloatTensor(shape=torch.Size((3, 3)))
    assert isinstance(a.__str__(), str)


@pytest.mark.parametrize("cmd", ["__add__", "sub", "__mul__"])
def test_operations_between_promises(hook, cmd):
    hook.local_worker.is_client_worker = False

    a = syft.Promises.FloatTensor(shape=torch.Size((2, 2)))
    b = syft.Promises.FloatTensor(shape=torch.Size((2, 2)))

    actual = getattr(a, cmd)(b)

    ta = torch.tensor([[1.0, 2], [3, 4]])
    tb = torch.tensor([[-8.0, -7], [6, 5]])
    a.keep(ta)
    b.keep(tb)

    expected = getattr(ta, cmd)(tb)

    assert (actual.value() == expected).all()

    hook.local_worker.is_client_worker = True


@pytest.mark.parametrize("cmd", ["__add__", "sub", "__mul__"])
def test_operations_with_concrete(hook, cmd):
    hook.local_worker.is_client_worker = False

    a = syft.Promises.FloatTensor(shape=torch.Size((2, 2)))
    b = torch.tensor([[-8.0, -7], [6, 5]]).wrap()  # TODO fix need to wrap

    actual = getattr(a, cmd)(b)

    ta = torch.tensor([[1.0, 2], [3, 4]]).wrap()
    a.keep(ta)

    expected = getattr(ta, cmd)(b)

    assert (actual.value() == expected).all()

    hook.local_worker.is_client_worker = True


def test_send(workers):
    bob = workers["bob"]

    a = syft.Promises.FloatTensor(shape=torch.Size((2, 2)))

    x = a.send(bob)

    x.keep(torch.ones((2, 2)))

    assert (x.value().get() == torch.ones((2, 2))).all()

    """
    # Send plans
    @syft.func2plan(args_shape=[(3,3)])
    def plan_test(data):
        return 2 * data + 1

    a = syft.Promises.FloatTensor(shape=torch.Size((3,3)))
    x = a.send(bob)
    
    ptr_plan = plan_test.send(bob)
    z = ptr_plan(x)

    a.keep(torch.ones(3,3))
    
    assert (z.value().get() == 3 * torch.ones(3, 3)).all()
    """


@pytest.mark.parametrize("cmd", ["__add__", "sub", "__mul__"])
def test_remote_operations(workers, cmd):
    bob = workers["bob"]

    a = syft.Promises.FloatTensor(shape=torch.Size((3, 3)))
    b = syft.Promises.FloatTensor(shape=torch.Size((3, 3)))

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

    a = syft.Promises.FloatTensor(shape=torch.Size((3, 3)))

    a.keep(torch.ones(3, 3))
    a.keep(2 * torch.ones(3, 3))
    a.keep(3 * torch.ones(3, 3))

    assert (a.value() == torch.ones(3, 3)).all()
    assert (a.value() == 2 * torch.ones(3, 3)).all()
    assert (a.value() == 3 * torch.ones(3, 3)).all()

    hook.local_worker.is_client_worker = True


"""
def test_plan_waiting_promise(hook):
    hook.local_worker.is_client_worker = False

    @syft.func2plan(args_shape=[(3,3)])
    def plan_test(data):
        return 2 * data + 1

    a = syft.Promises.FloatTensor(shape=torch.Size((3,3)))
    
    z = plan_test(a)

    a.keep(torch.ones(3,3))
    
    assert (z.value() == 3 * torch.ones(3, 3)).all()

    hook.local_worker.is_client_worker = True


def test_plan_waiting_several_promises(hook):
    hook.local_worker.is_client_worker = False

    @syft.func2plan(args_shape=[(3,3), (3,3)])
    def plan_test(in_a, in_b):
        return in_a + in_b

    a = syft.Promises.FloatTensor(shape=torch.Size((3,3)))
    b = syft.Promises.FloatTensor(shape=torch.Size((3,3)))

    z = plan_test(a, b)

    a.keep(torch.ones(3,3))
    a.keep(3 * torch.ones(3,3))
    
    b.keep(2 * torch.ones(3,3))
    b.keep(4 * torch.ones(3,3))

    assert (z.value() == 3 * torch.ones(3, 3)).all()
    assert (z.value() == 7 * torch.ones(3, 3)).all()

    hook.local_worker.is_client_worker = True
"""
