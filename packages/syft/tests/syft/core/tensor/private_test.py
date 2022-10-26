# third party
import numpy as np
import pytest

# syft absolute
import syft as sy

# from syft.core.tensor.autodp.gamma_tensor import GammaTensor
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
    private = tensor.private(min_val=low, max_val=high, data_subjects="Optimus Prime")
    assert isinstance(private, Tensor)
    assert isinstance(private.child, PT)
    assert isinstance(private.child.min_vals, lra)
    assert isinstance(private.child.max_vals, lra)
    assert private.child.min_vals.shape == private.child.shape
    assert private.child.max_vals.shape == private.child.shape
    assert isinstance(private.child.data_subjects, np.ndarray)
    assert private.child.data_subjects.shape == private.child.shape
    assert len(private.child.data_subjects.sum()) == 1


def test_list(tensor: Tensor, low: int, high: int) -> None:
    private = tensor.private(min_val=low, max_val=high, data_subjects=["Optimus Prime"])
    assert isinstance(private, Tensor)
    assert isinstance(private.child, PT)
    assert isinstance(private.child.min_vals, lra)
    assert isinstance(private.child.max_vals, lra)
    assert private.child.min_vals.shape == private.child.shape
    assert private.child.max_vals.shape == private.child.shape
    assert isinstance(private.child.data_subjects, np.ndarray)
    assert private.child.data_subjects.shape == private.child.shape
    assert len(private.child.data_subjects.sum()) == 1


def test_tuple(tensor: Tensor, low: int, high: int) -> None:
    private = tensor.private(min_val=low, max_val=high, data_subjects=("Optimus Prime"))
    assert isinstance(private, Tensor)
    assert isinstance(private.child, PT)
    assert isinstance(private.child.min_vals, lra)
    assert isinstance(private.child.max_vals, lra)
    assert private.child.min_vals.shape == private.child.shape
    assert private.child.max_vals.shape == private.child.shape
    assert isinstance(private.child.data_subjects, np.ndarray)
    assert private.child.data_subjects.shape == private.child.shape
    assert len(private.child.data_subjects.sum()) == 1


def test_array(tensor: Tensor, low: int, high: int) -> None:
    private = tensor.private(
        min_val=low, max_val=high, data_subjects=np.array(["Optimus Prime"])
    )
    assert isinstance(private, Tensor)
    assert isinstance(private.child, PT)
    assert isinstance(private.child.min_vals, lra)
    assert isinstance(private.child.max_vals, lra)
    assert private.child.min_vals.shape == private.child.shape
    assert private.child.max_vals.shape == private.child.shape
    assert isinstance(private.child.data_subjects, np.ndarray)
    assert private.child.data_subjects.shape == private.child.shape
    assert len(private.child.data_subjects.sum()) == 1
