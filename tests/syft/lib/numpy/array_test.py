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
