import pytest
import torch as th
from os import name as os_name

if os_name != "nt":
    from syft.frameworks.crypten import jail


def add_tensors(tensor):  # pragma: no cover
    # torch here is from the jail
    t = torch.tensor(5)  # noqa: F821
    return t + tensor


@pytest.mark.skipif("os_name == 'nt'")
def test_simple_func():
    func = jail.JailRunner(func=add_tensors)
    t = th.tensor(2)
    expected = th.tensor(7)
    assert expected == func(t)


def import_socket():  # pragma: no cover
    import socket  # noqa: F401


@pytest.mark.skipif("os_name == 'nt'")
def test_import():
    func = jail.JailRunner(func=import_socket)
    with pytest.raises(ImportError):
        func()


@pytest.mark.skipif("os_name == 'nt'")
def test_ser_deser():
    func = jail.JailRunner(func=add_tensors, modules=[th])
    func_ser = jail.JailRunner.simplify(func)
    func_deser = jail.JailRunner.detail(func_ser)

    t = th.tensor(2)
    expected = th.tensor(7)
    assert expected == func_deser(t)


@pytest.mark.skipif("os_name == 'nt'")
def test_empty_jail():
    with pytest.raises(ValueError) as e:
        empty_jail = jail.JailRunner()


@pytest.mark.parametrize(
    "src",
    [
        """
s = "not a function"
print(s)
        """,
        """
s = "not a function"
        """,
    ],
)
@pytest.mark.skipif("os_name == 'nt'")
def test_not_valid_func(src):
    with pytest.raises(ValueError):
        not_valid = jail.JailRunner(func_src=src)


@pytest.mark.skipif("os_name == 'nt'")
def test_already_built():
    func = jail.JailRunner(func=add_tensors)
    with pytest.raises(RuntimeWarning):
        func._build()
