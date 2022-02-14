# third party
import numpy as np

# syft absolute
from syft.core.tensor.passthrough import PassthroughTensor


def test_data_child() -> None:
    data = np.array([1, 2, 3], dtype=np.int32)
    tensor = PassthroughTensor(child=data)

    assert (tensor._data_child == data).all()
