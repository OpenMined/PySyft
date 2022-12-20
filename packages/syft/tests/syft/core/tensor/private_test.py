# third party
import numpy as np
import pytest

# syft absolute
import syft as sy
from syft.core.adp.data_subject_list import DataSubjectArray
from syft.core.adp.data_subject import DataSubject
from syft.core.tensor.autodp.gamma_tensor import GammaTensor as GT
from syft.core.tensor.autodp.phi_tensor import PhiTensor as PT
from syft.core.tensor.lazy_repeat_array import lazyrepeatarray as lra
from syft.core.tensor.tensor import Tensor


@pytest.fixture
def high() -> int:
    return 10


@pytest.fixture
def low() -> int:
    return 0


@pytest.fixture
def data(high: int, low: int) -> np.ndarray:
    return np.random.randint(low=low, high=high, size=(5, 5))


@pytest.fixture
def tensor(data: np.ndarray) -> Tensor:
    return sy.Tensor(data)


def test_string(tensor: Tensor, low: int, high: int) -> None:
    private = tensor.annotate_with_dp_metadata(
        lower_bound=low, upper_bound=high, data_subject="Optimus Prime"
    )
    assert isinstance(private, Tensor)
    assert isinstance(private.child, PT)
    assert isinstance(private.child.min_vals, lra)
    assert isinstance(private.child.max_vals, lra)
    assert private.child.min_vals.shape == private.child.shape
    assert private.child.max_vals.shape == private.child.shape
    assert isinstance(private.child.data_subject, DataSubject)


# def test_list(tensor: Tensor, low: int, high: int) -> None:
#     private = tensor.annotate_with_dp_metadata(
#         lower_bound=low, upper_bound=high, data_subjects=["Optimus Prime"]
#     )
#     assert isinstance(private, Tensor)
#     assert isinstance(private.child, PT)
#     assert isinstance(private.child.min_vals, lra)
#     assert isinstance(private.child.max_vals, lra)
#     assert private.child.min_vals.shape == private.child.shape
#     assert private.child.max_vals.shape == private.child.shape
#     assert isinstance(private.child.data_subjects, np.ndarray)
#     assert private.child.data_subjects.shape == private.child.shape
#     assert len(private.child.data_subjects.sum()) == 1


# def test_tuple(tensor: Tensor, low: int, high: int) -> None:
#     private = tensor.annotate_with_dp_metadata(
#         lower_bound=low, upper_bound=high, data_subjects=("Optimus Prime",)
#     )
#     assert isinstance(private, Tensor)
#     assert isinstance(private.child, PT)
#     assert isinstance(private.child.min_vals, lra)
#     assert isinstance(private.child.max_vals, lra)
#     assert private.child.min_vals.shape == private.child.shape
#     assert private.child.max_vals.shape == private.child.shape
#     assert isinstance(private.child.data_subjects, np.ndarray)
#     assert private.child.data_subjects.shape == private.child.shape
#     assert len(private.child.data_subjects.sum()) == 1


# def test_array(tensor: Tensor, low: int, high: int) -> None:
#     private = tensor.annotate_with_dp_metadata(
#         lower_bound=low, upper_bound=high, data_subjects=np.array(["Optimus Prime"])
#     )
#     assert isinstance(private, Tensor)
#     assert isinstance(private.child, PT)
#     assert isinstance(private.child.min_vals, lra)
#     assert isinstance(private.child.max_vals, lra)
#     assert private.child.min_vals.shape == private.child.shape
#     assert private.child.max_vals.shape == private.child.shape
#     assert isinstance(private.child.data_subjects, np.ndarray)
#     assert private.child.data_subjects.shape == private.child.shape
#     assert len(private.child.data_subjects.sum()) == 1


# def test_1d_list(tensor: Tensor, low: int, high: int) -> None:
#     private = tensor.annotate_with_dp_metadata(
#         lower_bound=low, upper_bound=high, data_subjects=["Optimus Prime"] * 5
#     )
#     assert isinstance(private, Tensor)
#     assert isinstance(private.child, PT)
#     assert isinstance(private.child.min_vals, lra)
#     assert isinstance(private.child.max_vals, lra)
#     assert private.child.min_vals.shape == private.child.shape
#     assert private.child.max_vals.shape == private.child.shape
#     assert isinstance(private.child.data_subjects, np.ndarray)
#     assert private.child.data_subjects.shape == private.child.shape
#     assert len(private.child.data_subjects.sum()) == 1


# def test_1d_tuple(tensor: Tensor, low: int, high: int) -> None:
#     private = tensor.annotate_with_dp_metadata(
#         lower_bound=low, upper_bound=high, data_subjects=tuple(["Optimus Prime"] * 5)
#     )
#     assert isinstance(private, Tensor)
#     assert isinstance(private.child, PT)
#     assert isinstance(private.child.min_vals, lra)
#     assert isinstance(private.child.max_vals, lra)
#     assert private.child.min_vals.shape == private.child.shape
#     assert private.child.max_vals.shape == private.child.shape
#     assert isinstance(private.child.data_subjects, np.ndarray)
#     assert private.child.data_subjects.shape == private.child.shape
#     assert len(private.child.data_subjects.sum()) == 1


# def test_1d_array(tensor: Tensor, low: int, high: int) -> None:
#     private = tensor.annotate_with_dp_metadata(
#         lower_bound=low, upper_bound=high, data_subjects=np.array(["Optimus Prime"] * 5)
#     )
#     assert isinstance(private, Tensor)
#     assert isinstance(private.child, PT)
#     assert isinstance(private.child.min_vals, lra)
#     assert isinstance(private.child.max_vals, lra)
#     assert private.child.min_vals.shape == private.child.shape
#     assert private.child.max_vals.shape == private.child.shape
#     assert isinstance(private.child.data_subjects, np.ndarray)
#     assert private.child.data_subjects.shape == private.child.shape
#     assert len(private.child.data_subjects.sum()) == 1


# def test_2d_list(tensor: Tensor, low: int, high: int) -> None:
#     private = tensor.annotate_with_dp_metadata(
#         lower_bound=low, upper_bound=high, data_subjects=[["Optimus Prime"] * 5] * 5
#     )
#     assert isinstance(private, Tensor)
#     assert isinstance(private.child, PT)
#     assert isinstance(private.child.min_vals, lra)
#     assert isinstance(private.child.max_vals, lra)
#     assert private.child.min_vals.shape == private.child.shape
#     assert private.child.max_vals.shape == private.child.shape
#     assert isinstance(private.child.data_subjects, np.ndarray)
#     assert private.child.data_subjects.shape == private.child.shape
#     assert len(private.child.data_subjects.sum()) == 1


# def test_2d_array(tensor: Tensor, low: int, high: int) -> None:
#     private = tensor.annotate_with_dp_metadata(
#         lower_bound=low,
#         upper_bound=high,
#         data_subjects=np.random.choice(["Optimus Prime"], (5, 5)),
#     )
#     assert isinstance(private, Tensor)
#     assert isinstance(private.child, PT)
#     assert isinstance(private.child.min_vals, lra)
#     assert isinstance(private.child.max_vals, lra)
#     assert private.child.min_vals.shape == private.child.shape
#     assert private.child.max_vals.shape == private.child.shape
#     assert isinstance(private.child.data_subjects, np.ndarray)
#     assert private.child.data_subjects.shape == private.child.shape
#     assert len(private.child.data_subjects.sum()) == 1


# def test_phi(tensor: Tensor, low: int, high: int) -> None:
#     data_subjects = np.random.choice(["Optimus Prime", "Bumblebee"], (5, 5))
#     # Make sure there's at least one of "Optimus Prime" and "Bumblebee" to prevent
#     # the 1/2^24 chance of failure
#     data_subjects[0, 0] = "Optimus Prime"
#     data_subjects[4, 4] = "Bumblebee"

#     private = tensor.private(
#         min_val=low,
#         max_val=high,
#         data_subjects=data_subjects,
#     )
#     assert isinstance(private, Tensor)
#     assert isinstance(private.child, PT)
#     assert isinstance(private.child.min_vals, lra)
#     assert isinstance(private.child.max_vals, lra)
#     assert private.child.min_vals.shape == private.child.shape
#     assert private.child.max_vals.shape == private.child.shape
#     assert isinstance(private.child.data_subjects, np.ndarray)
#     assert private.child.data_subjects.shape == private.child.shape
#     assert len(private.child.data_subjects.sum()) == 2


# def test_gamma(tensor: Tensor, low: int, high: int) -> None:
#     data_subjects = np.random.choice(["Optimus Prime", "Bumblebee"], (5, 5)).tolist()
#     data_subjects = [[DataSubjectArray([x]) for x in row] for row in data_subjects]
#     data_subjects[0][0] = DataSubjectArray(["Optimus Prime", "Bumblebee"])

#     private = tensor.annotate_with_dp_metadata(
#         lower_bound=low,
#         upper_bound=high,
#         data_subjects=data_subjects,
#     )
#     assert isinstance(private, Tensor)
#     assert isinstance(private.child, GT)
#     assert isinstance(private.child.min_vals, lra)
#     assert isinstance(private.child.max_vals, lra)
#     assert private.child.min_vals.shape == private.child.shape
#     assert private.child.max_vals.shape == private.child.shape
#     assert isinstance(private.child.data_subjects, np.ndarray)
#     assert private.child.data_subjects.shape == private.child.shape
#     assert len(private.child.data_subjects.sum()) == 2


# def test_repeat_list_arg(tensor: Tensor, low: int, high: int) -> None:
#     # https://github.com/OpenMined/PySyft/issues/6940
#     original = tensor.annotate_with_dp_metadata(
#         lower_bound=low,
#         upper_bound=high,
#         data_subjects=np.random.choice(["Optimus Prime"], (5, 5)),
#     )

#     private = original.repeat([1] * 5, axis=1)

#     assert private.shape == original.shape

#     assert isinstance(private, Tensor)
#     assert isinstance(private.child, PT)
#     assert isinstance(private.child.min_vals, lra)
#     assert isinstance(private.child.max_vals, lra)
#     assert private.child.min_vals.shape == private.child.shape
#     assert private.child.max_vals.shape == private.child.shape
#     assert isinstance(private.child.data_subjects, np.ndarray)
#     assert private.child.data_subjects.shape == private.child.shape
#     assert len(private.child.data_subjects.sum()) == 1
