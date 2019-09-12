import pytest
import torch
import syft


def test__str__():
    a = syft.Promises.FloatTensor(shape=torch.Size((3,3)))
    assert isinstance(a.__str__(), str)


@pytest.mark.parametrize("cmd", ["__add__", "sub", "__mul__"])
def test_operations_between_promises(cmd):
    a = syft.Promises.FloatTensor(shape=torch.Size((2,2)))
    b = syft.Promises.FloatTensor(shape=torch.Size((2,2)))

    actual = getattr(a, cmd)(b)
    
    a.keep(torch.tensor([[1., 2], [3, 4]]))
    b.keep(torch.tensor([[-8., -7], [6, 5]]))

    expected = getattr(a, cmd)(b)

    assert (actual == expected).all()


@pytest.mark.parametrize("cmd", ["__add__", "sub", "__mul__"])
def test_operations_with_concrete(cmd):
    a = syft.Promises.FloatTensor(shape=torch.Size((2,2)))
    b = torch.tensor([[-8., -7], [6, 5]]).wrap()  #TODO fix need to wrap

    actual = getattr(a, cmd)(b)
    
    a.keep(torch.tensor([[1., 2], [3, 4]]))

    expected = getattr(a, cmd)(b.child)

    assert (actual == expected).all()


# Still doesn't work
"""
def test_send(workers):
    bob = workers["bob"]

    a = syft.Promises.FloatTensor(shape=torch.Size((2,2)))
    
    x = a.send(bob)

    x.keep(torch.ones((2,2)))

    assert (x.get() == torch.ones((2,2))).all()


@pytest.mark.parametrize("cmd", ["__add__", "sub", "__mul__"])
def test_remote_operations(workers, cmd):
    a = sy.Promises.FloatTensor(shape=torch.Size((3,3)))
    b = sy.Promises.FloatTensor(shape=torch.Size((3,3)))

    x = a.send(bob)
    y = b.send(bob)

    actual = getattr(x, cmd)(y)

    x.keep(torch.tensor([[1., 2], [3, 4]]))
    y.keep(torch.tensor([[-8., -7], [6, 5]]))
    
    expected = getattr(a, cmd)(b)

    assert (actual == expected).all()
"""
