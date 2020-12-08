# stdlib
from typing import List as TypeList
from typing import Type as TypeType

# third party
from pydp.algorithms.laplacian import BoundedMean

# syft absolute
import syft as sy


def test_torch_function() -> None:
    bob = sy.VirtualMachine(name="Bob")
    client = bob.get_root_client()
    x_ptr = client.pydp.algorithms.laplacian.BoundedMean(1, 1, 50)
    
    input_data = [1, 88, 100, 5, 40, 30, 29, 56, 88, 23, 5, 1] * 100
    list_ptr = client.python.List(input_data)
    
    res_ptr = x_ptr.quick_result(list_ptr)
    res = res_ptr.get()

    assert (32 < res < 45)

