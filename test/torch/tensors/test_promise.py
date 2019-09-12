import pytest
import torch
import syft


def test__str__():
    a = syft.Promises.FloatTensor(shape=torch.Size((3,3)))
    assert isinstance(a.__str__(), str)


@pytest.mark.parametrize("cmd", ["__add__", "sub", "__mul__"])
def test_operations_between_promises(hook, cmd):
    hook.local_worker.is_client_worker = False
    
    a = syft.Promises.FloatTensor(shape=torch.Size((2,2)))
    b = syft.Promises.FloatTensor(shape=torch.Size((2,2)))

    actual = getattr(a, cmd)(b)
    
    ta = torch.tensor([[1., 2], [3, 4]])
    tb = torch.tensor([[-8., -7], [6, 5]])
    a.keep(ta)
    b.keep(tb)

    expected = getattr(ta, cmd)(tb)

    assert (actual.value() == expected).all()

    hook.local_worker.is_client_worker = True


@pytest.mark.parametrize("cmd", ["__add__", "sub", "__mul__"])
def test_operations_with_concrete(hook, cmd):
    hook.local_worker.is_client_worker = False
    
    a = syft.Promises.FloatTensor(shape=torch.Size((2,2)))
    b = torch.tensor([[-8., -7], [6, 5]]).wrap()  #TODO fix need to wrap

    actual = getattr(a, cmd)(b)
    
    ta = torch.tensor([[1., 2], [3, 4]])
    a.keep(ta)

    expected = getattr(ta, cmd)(b.child)

    assert (actual.value() == expected).all()

    hook.local_worker.is_client_worker = True


# Still doesn't work
def test_send(workers):
    bob = workers["bob"]

    a = syft.Promises.FloatTensor(shape=torch.Size((2,2)))
    
    x = a.send(bob)

    x.keep(torch.ones((2,2)))

    assert (x.value().get() == torch.ones((2,2))).all()


@pytest.mark.parametrize("cmd", ["__add__", "sub", "__mul__"])
def test_remote_operations(workers, cmd):
    bob = workers["bob"]

    a = syft.Promises.FloatTensor(shape=torch.Size((3,3)))
    b = syft.Promises.FloatTensor(shape=torch.Size((3,3)))

    x = a.send(bob)
    y = b.send(bob)

    actual = getattr(x, cmd)(y)

    tx = torch.tensor([[1., 2], [3, 4]])
    ty = torch.tensor([[-8., -7], [6, 5]])
    x.keep(tx)
    y.keep(ty)
    
    expected = getattr(tx, cmd)(ty)

    assert (actual.value().get() == expected).all()
