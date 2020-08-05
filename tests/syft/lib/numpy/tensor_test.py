import syft as sy
import numpy as np


def test_basic_numpy_array_serde():
    """Test that basic serialization and deserialization of numpy arrays works as expected"""

    x = np.array([1, 2, 3])
    proto_x = sy.serialize(obj=x)
    x2 = sy.deserialize(blob=proto_x)
    assert (x == x2).all()
