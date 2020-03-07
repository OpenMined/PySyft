import syft.numpy_ as np_
import numpy as np


def test_array_constructor():
    x = np.array([1, 2, 3, 4])

    assert isinstance(x, np.ndarray)

    assert (x == np.array([1, 2, 3, 4])).all()