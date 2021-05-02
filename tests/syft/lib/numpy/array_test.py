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


@pytest.mark.vendor(lib="numpy")
def test_strides(root_client: sy.VirtualMachineClient) -> None:
    # third party
    import numpy as np

    x = np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=np.int32)
    x_ptr = x.send(root_client)
    strides_ptr = x_ptr.strides
    local_strides_val = x.strides
    strides_val = strides_ptr.get()
    assert strides_val == (20, 4)
    assert local_strides_val == strides_val


@pytest.mark.vendor(lib="numpy")
def test_ndim(root_client: sy.VirtualMachineClient) -> None:
    # third party
    import numpy as np

    x = np.zeros((2, 3, 4))
    x_ptr = x.send(root_client)
    ndim_ptr = x_ptr.ndim
    local_ndim_val = x.ndim
    ndim_val = ndim_ptr.get()
    assert ndim_val == 3
    assert local_ndim_val == ndim_val


@pytest.mark.vendor(lib="numpy")
def test_size(root_client: sy.VirtualMachineClient) -> None:
    # third party
    import numpy as np

    x = np.zeros((3, 5, 2))
    x_ptr = x.send(root_client)
    size_ptr = x_ptr.size
    local_size_val = x.size
    size_val = size_ptr.get()
    assert size_val == 30
    assert local_size_val == size_val


@pytest.mark.vendor(lib="numpy")
def test_itemsize(root_client: sy.VirtualMachineClient) -> None:
    # third party
    import numpy as np

    x = np.array([1, 2, 3], dtype=np.float64)
    x_ptr = x.send(root_client)
    itemsize_ptr = x_ptr.itemsize
    local_itemsize_val = x.itemsize
    itemsize_val = itemsize_ptr.get()
    assert itemsize_val == 8
    assert local_itemsize_val == itemsize_val


@pytest.mark.vendor(lib="numpy")
def test_nbytes(root_client: sy.VirtualMachineClient) -> None:
    # third party
    import numpy as np

    x = np.zeros((3, 5, 2))
    x_ptr = x.send(root_client)
    nbytes_ptr = x_ptr.nbytes
    local_nbytes_val = x.nbytes
    nbytes_val = nbytes_ptr.get()
    assert nbytes_val == 240
    assert local_nbytes_val == nbytes_val


@pytest.mark.vendor(lib="numpy")
def test_base(root_client: sy.VirtualMachineClient) -> None:
    # third party
    import numpy as np

    x = np.array([1, 2, 3, 4])
    x_ptr = x.send(root_client)
    base_ptr = x_ptr.base
    base_val = base_ptr.get()
    assert base_val is None
