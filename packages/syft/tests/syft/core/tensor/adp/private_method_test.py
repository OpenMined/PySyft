# third party
import numpy as np

# syft absolute
import syft as sy
from syft.core.adp.entity import Entity
from syft.core.tensor.config import DEFAULT_INT_NUMPY_TYPE


def test_incompatible_input_tensor_type() -> None:

    try:
        x = sy.Tensor(np.float32([1, 2, 3, 4.0]))
        x.private(min_val=0, max_val=5, data_subjects="bob")
        raise AssertionError()
    except TypeError:
        assert True


def test_string_entity() -> None:
    x = sy.Tensor(np.array([1, 2, 3, 4], dtype=DEFAULT_INT_NUMPY_TYPE))
    out = x.private(min_val=0, max_val=5, data_subjects="bob")
    assert out.child.entity.name == "bob"


def test_list_of_strings_entity() -> None:
    x = sy.Tensor(np.array([1, 2, 3, 4], dtype=DEFAULT_INT_NUMPY_TYPE))
    out = x.private(min_val=0, max_val=5, data_subjects=["bob", "bob", "bob", "alice"])
    assert out.child.data_subjects[0][0] == "bob"
    assert out.child.data_subjects[-1][0] == "alice"


def test_class_entity() -> None:
    x = sy.Tensor(np.array([1, 2, 3, 4], dtype=DEFAULT_INT_NUMPY_TYPE))
    out = x.private(min_val=0, max_val=5, data_subjects=Entity("bob"))
    assert out.child.entity.name == "bob"


def test_list_of_entity_objs() -> None:
    x = sy.Tensor(np.array([1, 2, 3, 4], dtype=DEFAULT_INT_NUMPY_TYPE))
    out = x.private(
        min_val=0,
        max_val=5,
        data_subjects=[Entity("bob"), Entity("bob"), Entity("bob"), Entity("alice")],
    )
    assert out.child.data_subjects[0][0].name == "bob"
    assert out.child.data_subjects[-1][0].name == "alice"
