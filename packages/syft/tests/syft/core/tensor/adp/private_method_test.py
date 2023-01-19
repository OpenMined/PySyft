# third party
import numpy as np
import pytest

# syft absolute
import syft as sy
from syft.core.adp.data_subject_list import DataSubjectArray
from syft.core.tensor.config import DEFAULT_INT_NUMPY_TYPE

# def test_incompatible_input_tensor_type() -> None:
#     with pytest.raises(Exception):
#         x = sy.Tensor(np.float32([1, 2, 3, 4.0]))
#         bob = DataSubjectArray.from_objs(np.array(["bob", "billy"]))
#         x.annotate_with_dp_metadata(lower_bound=0, upper_bound=5, data_subject=bob)


def test_string_data_subject() -> None:
    x = sy.Tensor(np.array([1], dtype=DEFAULT_INT_NUMPY_TYPE))
    # bob = DataSubjectArray(["bob"])
    bob = "bob"
    # data_subjects = np.broadcast_to(np.array(bob), x.child.shape)
    out = x.annotate_with_dp_metadata(lower_bound=0, upper_bound=5, data_subject=bob)
    assert bob == out.child.data_subject


# def test_list_of_strings_data_subject() -> None:
#     x = sy.Tensor(np.array([1, 2, 3, 4], dtype=DEFAULT_INT_NUMPY_TYPE))
#     bob = DataSubjectArray(["bob"])
#     alice = DataSubjectArray(["alice"])
#     data_subjects = np.array([alice, alice, alice, bob])
#     out = x.annotate_with_dp_metadata(
#         lower_bound=0, upper_bound=5, data_subjects=data_subjects
#     )
#     assert bob in out.child.data_subjects
#     assert alice in out.child.data_subjects
