# syft absolute
import syft as sy


def test_pydp() -> None:
    bob = sy.VirtualMachine(name="Bob")
    client = bob.get_root_client()
    x_ptr = client.pydp.algorithms.laplacian.BoundedMean(1, 1, 50)

    input_data = [1, 88, 100, 5, 40, 30, 29, 56, 88, 23, 5, 1] * 100
    list_ptr = client.python.List(input_data)

    res_ptr = x_ptr.quick_result(list_ptr)
    res = res_ptr.get()

    assert 22 < res < 35


def test_pydp_functions() -> None:
    bob = sy.VirtualMachine(name="Bob")
    client = bob.get_root_client()
    x_ptr = client.pydp.algorithms.laplacian.BoundedMean(1, 1, 50)

    input_data = [1, 88, 100, 5, 40, 30, 29, 56, 88, 23, 5, 1] * 100
    list_ptr = client.python.List(input_data)
    x_ptr.add_entries(list_ptr)
    res_ptr = x_ptr.result(0.7)
    assert isinstance(res_ptr.get(), float)

    assert round(x_ptr.privacy_budget_left().get(), 2) == 0.3
