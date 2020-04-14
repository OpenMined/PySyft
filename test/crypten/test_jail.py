import torch as th
from syft.frameworks.crypten import jail


def add_tensors(tensor):
    # torch here is from the jail
    t = torch.tensor(5)  # noqa
    return t + tensor


def test_simple_func():
    func = jail.JailRunner(func=add_tensors)
    t = th.tensor(2)
    expected = th.tensor(7)
    assert expected == func(t)


def import_socket():
    import socket


def test_import():
    func = jail.JailRunner(func=import_socket)
    catched = True

    try:
        func()
        catched = False
    except Exception as e:
        assert isinstance(e, ImportError)

    assert catched
