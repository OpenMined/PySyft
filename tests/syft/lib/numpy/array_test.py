# stdlib
from typing import List

# third party
import pytest

# syft absolute
import syft as sy


@pytest.mark.vendor(lib="numpy")
def test_remote_numpy_array(root_client: sy.VirtualMachineClient) -> None:
    # third party
    import numpy as np

    # syft absolute
    from syft.lib.numpy.array import SUPPORTED_BOOL_TYPES
    from syft.lib.numpy.array import SUPPORTED_DTYPES
    from syft.lib.numpy.array import SUPPORTED_FLOAT_TYPES
    from syft.lib.numpy.array import SUPPORTED_INT_TYPES

    sy.load("numpy")

    test_arrays: List[np.ndarray] = []
    for dtype in SUPPORTED_DTYPES:

        # test their bounds
        if dtype in SUPPORTED_BOOL_TYPES:
            lower = False
            upper = True
            mid = False
        elif dtype in SUPPORTED_INT_TYPES:
            bounds = np.iinfo(dtype)
            lower = bounds.min
            upper = bounds.max
            mid = upper + lower  # type: ignore
            if lower == 0:
                mid = round(mid / 2)  # type: ignore
        elif dtype in SUPPORTED_FLOAT_TYPES:
            bounds = np.finfo(dtype)
            lower = bounds.min
            upper = bounds.max
            mid = upper + lower  # type: ignore

        test_arrays.append(np.array([lower, mid, upper], dtype=dtype))

    for test_array in test_arrays:
        remote_array = test_array.send(root_client)
        received_array = remote_array.get()

        assert all(test_array == received_array)
        assert test_array.dtype == received_array.dtype

# Attributes test

@pytest.mark.vendor(lib="numpy")
def test_flags(root_client: sy.VirtualMachineClient) -> None:
    # third party
    import numpy as np
    x = np.array([1, 2, 3, 4])
    x_ptr = x.send(root_client)
    flags_ptr = x_ptr.flags
    local_flags = x.flags
    flags_val = flags_ptr.get()
    assert flags_val == x.writeable
    assert local_flags == flags_val

@pytest.mark.vendor(lib="numpy")
def test_shape(root_client: sy.VirtualMachineClient) -> None:
    # third party
    import numpy as np

    x = np.array([1, 2, 3, 4])
    x_ptr = x.send(root_client)
    shape_ptr = x_ptr.shape
    local_shape_val = x.shape
    shape_val = shape_ptr.get()
    assert shape_val == (4,)
    assert local_shape_val == shape_val