# third party
import numpy as np
import pytest

# syft absolute
import syft as sy
from syft.core.adp.data_subject_list import DataSubjectArray
from syft.core.tensor.config import DEFAULT_INT_NUMPY_TYPE


def test_incompatible_input_tensor_type() -> None:
    with pytest.raises(ValueError):
        x = sy.Tensor(np.float32([1, 2, 3, 4.0]))
        bob = DataSubjectArray(["bob"])
        x.private(min_val=0, max_val=5, data_subjects=bob)


def test_string_data_subject() -> None:
    x = sy.Tensor(np.array([1], dtype=DEFAULT_INT_NUMPY_TYPE))
    bob = DataSubjectArray(["bob"])
    data_subjects = np.broadcast_to(np.array(bob), x.child.shape)
    out = x.private(min_val=0, max_val=5, data_subjects=data_subjects)
    assert bob in out.child.data_subjects


def test_list_of_strings_data_subject() -> None:
    x = sy.Tensor(np.array([1, 2, 3, 4], dtype=DEFAULT_INT_NUMPY_TYPE))
    bob = DataSubjectArray(["bob"])
    alice = DataSubjectArray(["alice"])
    data_subjects = np.array([alice, alice, alice, bob])
    out = x.private(min_val=0, max_val=5, data_subjects=data_subjects)
    assert bob in out.child.data_subjects
    assert alice in out.child.data_subjects
