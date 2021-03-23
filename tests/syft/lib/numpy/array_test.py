# third party
import numpy as np
import pytest

# syft absolute
import syft as sy

ExampleArray = [
    np.array([1, 2, -3], dtype=np.int8),
    np.array([1, 2, -3], dtype=np.int16),
    np.array([1, 2, -3], dtype=np.int32),
    np.array([1, 2, -3], dtype=np.int64),
    np.array([1, 2, 3], dtype=np.uint8),
    # np.array([1, 2, 3], dtype=np.uint16),
    # np.array([1, 2, 3], dtype=np.uint32),
    # np.array([1, 2, 3], dtype=np.uint64),
    np.array([1.2, 2.2, 3.0], dtype=np.float16),
    np.array([1.2, 2.2, 3.0], dtype=np.float32),
    np.array([1.2, 2.2, 3.0], dtype=np.float64),
    # np.array([1 + 2j, 3 + 4j, 5 + 0j], dtype=np.complex64),
    np.array([True, False, True], dtype=np.bool_),
]


@pytest.mark.vendor(lib="numpy")
def test_remote_numpy_array() -> None:
    sy.load("numpy")

    vm = sy.VirtualMachine()
    client = vm.get_root_client()

    for test_array in ExampleArray:
        remote_array = test_array.send(client)
        received_array = remote_array.get()

        assert all(test_array == received_array)
