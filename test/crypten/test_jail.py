import pytest
import torch as th
from syft.frameworks.crypten import jail


def add_tensors(tensor):  # pragma: no cover
    # torch here is from the jail
    t = torch.tensor(5)  # noqa: F821
    return t + tensor


def test_simple_func():
    func = jail.JailRunner(func=add_tensors)
    t = th.tensor(2)
    expected = th.tensor(7)
    assert expected == func(t)


def import_socket():  # pragma: no cover
    import socket


def test_import():
    func = jail.JailRunner(func=import_socket)
    with pytest.raises(ImportError):
        func()


def test_ser_deser():
    func = jail.JailRunner(func=add_tensors, modules=[th])
    func_ser = jail.JailRunner.simplify(func)
    func_deser = jail.JailRunner.detail(func_ser)

    t = th.tensor(2)
    expected = th.tensor(7)
    assert expected == func_deser(t)
