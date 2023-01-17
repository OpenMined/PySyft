# stdlib
from typing import Any
from typing import Dict

# third party
from jax import numpy as jnp
import numpy as np
from numpy.typing import ArrayLike
import pytest

# syft absolute
import syft as sy
from syft.core.adp.data_subject_ledger import DataSubjectLedger
from syft.core.adp.data_subject_list import DataSubject
from syft.core.adp.ledger_store import DictLedgerStore
from syft.core.tensor.autodp.gamma_tensor import GammaTensor
from syft.core.tensor.autodp.phi_tensor import PhiTensor as PT
from syft.core.tensor.lazy_repeat_array import lazyrepeatarray as lra


@pytest.fixture
def ishan() -> ArrayLike:
    return np.array(DataSubject(["φhishan"]))


@pytest.fixture
def traskmaster() -> ArrayLike:
    return np.ndarray(DataSubject(["λamdrew"]))


@pytest.fixture
def highest() -> int:
    return 5


@pytest.fixture
def lowest(highest) -> int:
    return -1 * int(highest)


@pytest.fixture
def dims() -> int:
    """This generates a random integer for the number of dimensions in our testing tensors"""
    dims = int(max(3, np.random.randint(5) + 3))  # Avoid size 0 and 1
    # Failsafe
    if dims < 2:
        dims += 3
    assert dims > 1, "Tensor not large enough for several tests."
    return dims


@pytest.fixture
def reference_data(highest, dims) -> np.ndarray:
    """This generates random data to test the equality operators"""
    reference_data = np.random.randint(
        low=-highest, high=highest, size=(dims, dims), dtype=np.int64
    )
    assert dims > 1, "Tensor not large enough"
    return reference_data


@pytest.fixture
def dsa(dims: int) -> DataSubject:
    return DataSubject.from_objs(np.random.choice([0, 1], (dims, dims)))


@pytest.fixture
def upper_bound(reference_data: np.ndarray, highest: int) -> np.ndarray:
    """This is used to specify the max_vals that is either binary or randomly generated b/w 0-1"""
    return lra(data=highest, shape=reference_data.shape)


@pytest.fixture
def lower_bound(reference_data: np.ndarray, highest: int) -> np.ndarray:
    """This is used to specify the min_vals that is either binary or randomly generated b/w 0-1"""
    return lra(data=-highest, shape=reference_data.shape)


def test_gamma_serde(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
) -> None:
    """Test basic serde for GammaTensor"""
    data_subject = DataSubject(["eagle"])
    tensor1 = PT(
        child=reference_data,
        data_subject=data_subject,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )
    gamma_tensor1 = tensor1.gamma
    
    # Checks to ensure gamma tensor was properly created
    assert isinstance(gamma_tensor1, GammaTensor)
    assert (gamma_tensor1.child == tensor1.child).all()

    ser = sy.serialize(gamma_tensor1, to_bytes=True)
    de = sy.deserialize(ser, from_bytes=True)

    assert (de.child == gamma_tensor1.child).all()
    assert de.is_linear == gamma_tensor1.is_linear

    de_state = {}
    for key in de.sources:
        de_state[key] = de.sources[key].child
    gamma_tensor1_state = {}
    for key in de.sources:
        gamma_tensor1_state[key] = de.sources[key].child

    assert (de.func(de_state) == gamma_tensor1.func(gamma_tensor1_state)).all()
    assert de.id == gamma_tensor1.id
    assert de.sources.keys() == gamma_tensor1.sources.keys()


def test_lipschitz(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubject,
) -> None:
    """Test lipschitz bound for GammaTensor"""
    tensor1 = PT(
        child=reference_data,
        data_subject=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )

    gamma_tensor = tensor1.gamma + 2
    assert gamma_tensor.is_linear
    assert gamma_tensor.lipschitz_bound == 1

    gamma_tensor = tensor1[0].gamma ** 2
    assert gamma_tensor.lipschitz_bound == 2 * max(tensor1[0].child)


def test_zeros_like(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubject,
) -> None:
    reference_tensor = PT(
        child=reference_data,
        data_subject=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )
    gamma_tensor = reference_tensor.gamma
    output = gamma_tensor.zeros_like()
    assert np.all(output.child == 0)
    assert list(output.sources.keys()) == [reference_tensor.id]
    state = {}
    for key in output.sources:
        state[key] = output.sources[key].child
    assert (output.func(state) == output.child).all()


def test_ones_like(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubject,
) -> None:
    reference_tensor = PT(
        child=reference_data,
        data_subject=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )

    gamma_tensor = reference_tensor.gamma
    output = gamma_tensor.ones_like()
    assert np.all(output.child == 1)
    assert list(output.sources.keys()) == [reference_tensor.id]
    state = {}
    for key in output.sources:
        state[key] = output.sources[key].child
    assert (output.func(state) == output.child).all()


def test_sum(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubject,
) -> None:
    reference_tensor = PT(
        child=reference_data,
        data_subject=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )

    gamma_tensor = reference_tensor.gamma

    result = gamma_tensor.sum()
    assert result.child == reference_data.sum()

    output = gamma_tensor.sum(axis=1)
    assert (output.child == reference_data.sum(axis=1)).all()

    assert list(output.sources.keys()) == [reference_tensor.id]
    state = {}
    for key in output.sources:
        state[key] = output.sources[key].child
    assert (output.func(state) == output.child).all()


def test_pow(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubject,
) -> None:
    tensor = PT(
        child=reference_data,
        data_subject=ishan,
        min_vals=lower_bound,
        max_vals=upper_bound,
    )
    gamma_tensor = tensor.gamma
    output = gamma_tensor.__pow__(2)
    assert (output.child == (reference_data**2)).all()

    assert list(output.sources.keys()) == [tensor.id]
    state = {}
    for key in output.sources:
        state[key] = output.sources[key].child
    assert (output.func(state) == output.child).all()


@pytest.mark.arithmetic
@pytest.mark.public_op
def test_add_public(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubject,
) -> None:
    ishan = np.broadcast_to(ishan, reference_data.shape)
    reference_tensor = PT(
        child=reference_data,
        data_subject=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )
    gamma_tensor = reference_tensor.gamma

    output = gamma_tensor + 5
    assert output.shape == reference_tensor.shape
    assert (output.child == reference_data + 5).all()

    assert list(output.sources.keys()) == [reference_tensor.id]
    state = {}
    for key in output.sources:
        state[key] = output.sources[key].child
    assert (output.func(state) == output.child).all()


@pytest.mark.arithmetic
def test_radd(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubject,
) -> None:
    reference_tensor = PT(
        child=reference_data,
        data_subject=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )
    gamma_tensor = reference_tensor.gamma
    input_data = np.zeros_like(reference_data)
    output = input_data + gamma_tensor
    assert output.shape == reference_tensor.shape
    assert (output.child == input_data + reference_data).all()

    assert list(output.sources.keys()) == [reference_tensor.id]
    state = {}
    for key in output.sources:
        state[key] = output.sources[key].child
    assert (output.func(state) == output.child).all()


@pytest.mark.arithmetic
def test_rsub(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubject,
) -> None:
    ishan = np.broadcast_to(ishan, reference_data.shape)
    reference_tensor = PT(
        child=reference_data,
        data_subject=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )
    gamma_tensor = reference_tensor.gamma
    input_data = np.zeros_like(reference_data)
    output = input_data - gamma_tensor
    assert output.shape == reference_tensor.shape
    assert (output.child == input_data - reference_data).all()
    assert list(output.sources.keys()) == [reference_tensor.id]
    state = {}
    for key in output.sources:
        state[key] = output.sources[key].child
    assert (output.func(state) == output.child).all()


@pytest.mark.arithmetic
def test_rmul(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubject,
) -> None:
    reference_tensor = PT(
        child=reference_data,
        data_subject=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )
    gamma_tensor = reference_tensor.gamma
    input_data = np.zeros_like(reference_data)
    output = input_data * gamma_tensor
    assert output.shape == reference_tensor.shape
    assert (output.child == input_data * reference_data).all()
    assert list(output.sources.keys()) == [reference_tensor.id]
    state = {}
    for key in output.sources:
        state[key] = output.sources[key].child
    assert (output.func(state) == output.child).all()


@pytest.mark.arithmetic
def test_rmatmul(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubject,
) -> None:
    reference_tensor = PT(
        child=reference_data,
        data_subject=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )
    gamma_tensor = reference_tensor.gamma
    input_data = np.zeros_like(reference_data)
    output = input_data @ gamma_tensor
    assert output.shape == reference_tensor.shape
    assert (output.child == input_data @ reference_data).all()

    assert list(output.sources.keys()) == [reference_tensor.id]
    state = {}
    for key in output.sources:
        state[key] = output.sources[key].child
    assert (output.func(state) == output.child).all()


@pytest.mark.arithmetic
def test_rtruediv(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubject,
) -> None:
    reference_tensor = PT(
        child=reference_data,
        data_subject=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )
    gamma_tensor = reference_tensor.gamma + 10  # we add 10 to avoid 0 values
    input_data = np.ones_like(reference_data)
    output = input_data / gamma_tensor
    assert output.shape == reference_tensor.shape
    assert (output.child == input_data / (reference_data + 10)).all()

    assert list(output.sources.keys()) == [reference_tensor.id]
    state = {}
    for key in output.sources:
        state[key] = output.sources[key].child
    assert (output.func(state) == output.child).all()


@pytest.mark.arithmetic
def test_rfloordiv(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubject,
) -> None:
    ishan = np.broadcast_to(ishan, reference_data.shape)
    reference_tensor = PT(
        child=reference_data,
        data_subject=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )
    gamma_tensor = reference_tensor.gamma + 10
    input_data = np.ones_like(reference_data)
    output = input_data // gamma_tensor
    assert output.shape == reference_tensor.shape
    assert (output.child == input_data // (reference_data + 10)).all()

    assert list(output.sources.keys()) == [reference_tensor.id]
    state = {}
    for key in output.sources:
        state[key] = output.sources[key].child
    assert (output.func(state) == output.child).all()


@pytest.mark.arithmetic
@pytest.mark.public_op
def test_sub_public(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubject,
) -> None:
    ishan = np.broadcast_to(ishan, reference_data.shape)
    reference_tensor = PT(
        child=reference_data,
        data_subject=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )

    gamma_tensor = reference_tensor.gamma
    output = gamma_tensor - 5
    assert output.shape == reference_tensor.shape
    assert (output.child == reference_data - 5).all()
    assert list(output.sources.keys()) == [reference_tensor.id]
    state = {}
    for key in output.sources:
        state[key] = output.sources[key].child
    assert (output.func(state) == output.child).all()


@pytest.mark.arithmetic
@pytest.mark.public_op
def test_mul_public(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubject,
) -> None:
    reference_tensor = PT(
        child=reference_data,
        data_subject=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )
    gamma_tensor = reference_tensor.gamma

    output = gamma_tensor * 5
    assert output.shape == reference_tensor.shape
    assert (output.child == reference_data * 5).all()
    assert list(output.sources.keys()) == [reference_tensor.id]
    state = {}
    for key in output.sources:
        state[key] = output.sources[key].child
    assert (output.func(state) == output.child).all()


@pytest.mark.arithmetic
@pytest.mark.public_op
def test_truediv_public(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubject,
) -> None:
    reference_tensor = PT(
        child=reference_data,
        data_subject=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )

    gamma_tensor = reference_tensor.gamma
    output = gamma_tensor / 5
    assert output.shape == reference_tensor.shape
    assert (output.child == reference_data / 5).all()
    assert list(output.sources.keys()) == [reference_tensor.id]
    state = {}
    for key in output.sources:
        state[key] = output.sources[key].child
    assert (output.func(state) == output.child).all()


@pytest.mark.arithmetic
@pytest.mark.public_op
def test_mod_public(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubject,
) -> None:
    ishan = np.broadcast_to(ishan, reference_data.shape)
    reference_tensor = PT(
        child=reference_data,
        data_subject=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )

    gamma_tensor = reference_tensor.gamma

    output = gamma_tensor % 5
    assert output.shape == reference_tensor.shape
    assert (output.child == reference_data % 5).all()
    assert list(output.sources.keys()) == [reference_tensor.id]
    state = {}
    for key in output.sources:
        state[key] = output.sources[key].child
    assert (output.func(state) == output.child).all()

    output = gamma_tensor % -5
    assert output.shape == reference_tensor.shape
    assert (output.child == reference_data % -5).all()
    assert list(output.sources.keys()) == [reference_tensor.id]
    state = {}
    for key in output.sources:
        state[key] = output.sources[key].child
    assert (output.func(state) == output.child).all()


@pytest.mark.arithmetic
@pytest.mark.public_op
def test_and_public(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubject,
) -> None:
    reference_tensor = PT(
        child=reference_data,
        data_subject=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )
    gamma_tensor = reference_tensor.gamma

    output = gamma_tensor & 5
    assert output.shape == reference_tensor.shape
    assert (output.child == reference_data & 5).all()
    assert list(output.sources.keys()) == [reference_tensor.id]
    state = {}
    for key in output.sources:
        state[key] = output.sources[key].child
    assert (output.func(state) == output.child).all()


@pytest.mark.arithmetic
@pytest.mark.public_op
def test_or_public(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubject,
) -> None:
    reference_tensor = PT(
        child=reference_data,
        data_subject=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )

    gamma_tensor = reference_tensor.gamma

    output = gamma_tensor | 5
    assert output.shape == reference_tensor.shape
    assert (output.child == reference_data | 5).all()
    assert list(output.sources.keys()) == [reference_tensor.id]
    state = {}
    for key in output.sources:
        state[key] = output.sources[key].child
    assert (output.func(state) == output.child).all()


@pytest.mark.arithmetic
@pytest.mark.private_op
def test_add_private(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubject,
) -> None:
    reference_tensor = PT(
        child=reference_data,
        data_subject=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )
    gamma_tensor1 = reference_tensor.gamma

    tensor2 = PT(
        child=reference_data,
        data_subject=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )
    gamma_tensor2 = tensor2.gamma

    output = gamma_tensor1 + gamma_tensor2
    assert output.shape == reference_tensor.shape
    assert (output.child == reference_data * 2).all()
    assert list(output.sources.keys()) == [reference_tensor.id, tensor2.id]
    state = {}
    for key in output.sources:
        state[key] = output.sources[key].child
    assert (output.func(state) == output.child).all()


@pytest.mark.arithmetic
@pytest.mark.private_op
def test_sub_private(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubject,
) -> None:
    reference_tensor = PT(
        child=reference_data,
        data_subject=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )
    gamma_tensor1 = reference_tensor.gamma

    tensor2 = PT(
        child=reference_data,
        data_subject=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )
    gamma_tensor2 = tensor2.gamma

    output = gamma_tensor1 - gamma_tensor2
    assert output.shape == reference_tensor.shape
    assert (output.child == 0).all()
    assert list(output.sources.keys()) == [reference_tensor.id, tensor2.id]
    state = {}
    for key in output.sources:
        state[key] = output.sources[key].child
    assert (output.func(state) == output.child).all()


@pytest.mark.arithmetic
@pytest.mark.private_op
def test_mul_private(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubject,
) -> None:
    reference_tensor = PT(
        child=reference_data,
        data_subject=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )
    gamma_tensor1 = reference_tensor.gamma

    tensor2 = PT(
        child=reference_data,
        data_subject=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )
    gamma_tensor2 = tensor2.gamma

    output = gamma_tensor1 * gamma_tensor2
    assert output.shape == reference_tensor.shape
    assert (output.child == reference_data**2).all()
    assert list(output.sources.keys()) == [reference_tensor.id, tensor2.id]
    state = {}
    for key in output.sources:
        state[key] = output.sources[key].child
    assert (output.func(state) == output.child).all()


@pytest.mark.arithmetic
@pytest.mark.private_op
def test_truediv_private(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubject,
) -> None:
    reference_tensor = PT(
        child=np.ones_like(reference_data),
        data_subject=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )
    gamma_tensor1 = reference_tensor.gamma

    tensor2 = PT(
        child=np.ones_like(reference_data),
        data_subject=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )
    gamma_tensor2 = tensor2.gamma

    output = gamma_tensor1 / gamma_tensor2
    assert output.shape == reference_tensor.shape
    assert (output.child == 1).all()
    assert list(output.sources.keys()) == [reference_tensor.id, tensor2.id]
    state = {}
    for key in output.sources:
        state[key] = output.sources[key].child
    assert (output.func(state) == output.child).all()


@pytest.mark.arithmetic
@pytest.mark.private_op
def test_mod_private(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubject,
) -> None:
    reference_tensor = PT(
        child=reference_data,
        data_subject=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )
    gamma_tensor1 = reference_tensor.gamma

    tensor2 = PT(
        child=reference_data,
        data_subject=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )
    gamma_tensor2 = tensor2.gamma

    output = gamma_tensor1 % gamma_tensor2
    assert output.shape == reference_tensor.shape
    assert (output.child == reference_data % reference_data).all()
    assert list(output.sources.keys()) == [reference_tensor.id, tensor2.id]
    state = {}
    for key in output.sources:
        state[key] = output.sources[key].child
    assert (output.func(state) == output.child).all()


@pytest.mark.arithmetic
@pytest.mark.private_op
def test_and_private(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubject,
) -> None:
    reference_tensor = PT(
        child=reference_data,
        data_subject=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )
    gamma_tensor1 = reference_tensor.gamma

    tensor2 = PT(
        child=reference_data,
        data_subject=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )
    gamma_tensor2 = tensor2.gamma

    output = gamma_tensor1 & gamma_tensor2
    assert output.shape == reference_tensor.shape
    assert (output.child == reference_data & reference_data).all()
    assert list(output.sources.keys()) == [reference_tensor.id, tensor2.id]
    state = {}
    for key in output.sources:
        state[key] = output.sources[key].child
    assert (output.func(state) == output.child).all()


@pytest.mark.arithmetic
@pytest.mark.private_op
def test_or_private(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubject,
) -> None:
    reference_tensor = PT(
        child=reference_data,
        data_subject=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )
    gamma_tensor1 = reference_tensor.gamma

    tensor2 = PT(
        child=reference_data,
        data_subject=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )
    gamma_tensor2 = tensor2.gamma

    output = gamma_tensor1 | gamma_tensor2
    assert output.shape == reference_tensor.shape
    assert (output.child == reference_data | reference_data).all()
    assert list(output.sources.keys()) == [reference_tensor.id, tensor2.id]
    state = {}
    for key in output.sources:
        state[key] = output.sources[key].child
    assert (output.func(state) == output.child).all()


@pytest.mark.equality
@pytest.mark.public_op
def test_eq_public(
    reference_data: np.ndarray,
    upper_bound: lra,
    lower_bound: lra,
    ishan: DataSubject,
) -> None:
    reference_tensor = PT(
        child=np.ones_like(reference_data),
        data_subject=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )
    gamma_tensor = reference_tensor.gamma

    # Test that it IS equal
    output = gamma_tensor == 1
    assert output.shape == reference_tensor.shape
    assert output.child.all()
    assert list(output.sources.keys()) == [reference_tensor.id]
    state = {}
    for key in output.sources:
        state[key] = output.sources[key].child
    assert (output.func(state) == output.child).all()


@pytest.mark.equality
@pytest.mark.public_op
def test_ne_public(
    reference_data: np.ndarray,
    upper_bound: lra,
    lower_bound: lra,
    ishan: DataSubject,
) -> None:
    reference_tensor = PT(
        child=np.ones_like(reference_data),
        data_subject=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )
    gamma_tensor = reference_tensor.gamma

    output = gamma_tensor != 0
    assert output.shape == reference_tensor.shape
    assert output.child.all()
    assert list(output.sources.keys()) == [reference_tensor.id]
    state = {}
    for key in output.sources:
        state[key] = output.sources[key].child
    assert (output.func(state) == output.child).all()


@pytest.mark.equality
@pytest.mark.public_op
def test_lt_public(
    reference_data: np.ndarray,
    upper_bound: lra,
    lower_bound: lra,
    ishan: DataSubject,
) -> None:
    reference_tensor = PT(
        child=np.ones_like(reference_data),
        data_subject=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )
    gamma_tensor = reference_tensor.gamma

    output = gamma_tensor < 2
    assert output.shape == reference_tensor.shape
    assert output.child.all()
    assert list(output.sources.keys()) == [reference_tensor.id]
    state = {}
    for key in output.sources:
        state[key] = output.sources[key].child
    assert (output.func(state) == output.child).all()


@pytest.mark.equality
@pytest.mark.public_op
def test_gt_public(
    reference_data: np.ndarray,
    upper_bound: lra,
    lower_bound: lra,
    ishan: DataSubject,
) -> None:
    ishan = np.broadcast_to(ishan, reference_data.shape)
    reference_tensor = PT(
        child=np.ones_like(reference_data),
        data_subject=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )
    gamma_tensor = reference_tensor.gamma

    output = gamma_tensor > 0
    assert output.shape == reference_tensor.shape
    assert output.child.all()
    assert list(output.sources.keys()) == [reference_tensor.id]
    state = {}
    for key in output.sources:
        state[key] = output.sources[key].child
    assert (output.func(state) == output.child).all()


@pytest.mark.equality
@pytest.mark.public_op
def test_le_public(
    reference_data: np.ndarray,
    upper_bound: lra,
    lower_bound: lra,
    ishan: DataSubject,
) -> None:
    reference_tensor = PT(
        child=np.ones_like(reference_data),
        data_subject=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )
    gamma_tensor = reference_tensor.gamma

    output = gamma_tensor <= 2
    assert output.shape == reference_tensor.shape
    assert output.child.all()
    assert list(output.sources.keys()) == [reference_tensor.id]
    state = {}
    for key in output.sources:
        state[key] = output.sources[key].child
    assert (output.func(state) == output.child).all()


@pytest.mark.equality
@pytest.mark.public_op
def test_ge_public(
    reference_data: np.ndarray,
    upper_bound: lra,
    lower_bound: lra,
    ishan: DataSubject,
) -> None:
    reference_tensor = PT(
        child=np.ones_like(reference_data),
        data_subject=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )
    gamma_tensor = reference_tensor.gamma

    output = gamma_tensor >= 0
    assert output.shape == reference_tensor.shape
    assert output.child.all()
    assert list(output.sources.keys()) == [reference_tensor.id]
    state = {}
    for key in output.sources:
        state[key] = output.sources[key].child
    assert (output.func(state) == output.child).all()


@pytest.mark.equality
@pytest.mark.private_op
def test_eq_private(
    reference_data: np.ndarray,
    upper_bound: lra,
    lower_bound: lra,
    ishan: DataSubject,
) -> None:
    reference_tensor = PT(
        child=np.ones_like(reference_data),
        data_subject=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )
    gamma_tensor = reference_tensor.gamma

    # Test that it IS equal
    output = gamma_tensor == gamma_tensor.ones_like()
    assert output.shape == reference_tensor.shape
    assert output.child.all()
    assert list(output.sources.keys()) == [reference_tensor.id]
    state = {}
    for key in output.sources:
        state[key] = output.sources[key].child
    assert (output.func(state) == output.child).all()


@pytest.mark.equality
@pytest.mark.private_op
def test_ne_private(
    reference_data: np.ndarray,
    upper_bound: lra,
    lower_bound: lra,
    ishan: DataSubject,
) -> None:
    reference_tensor = PT(
        child=np.ones_like(reference_data),
        data_subject=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )
    gamma_tensor = reference_tensor.gamma

    output = gamma_tensor != gamma_tensor.zeros_like()
    assert output.shape == reference_tensor.shape
    assert output.child.all()
    assert list(output.sources.keys()) == [reference_tensor.id]
    state = {}
    for key in output.sources:
        state[key] = output.sources[key].child
    assert (output.func(state) == output.child).all()


@pytest.mark.equality
@pytest.mark.private_op
def test_lt_private(
    reference_data: np.ndarray,
    upper_bound: lra,
    lower_bound: lra,
    ishan: DataSubject,
) -> None:
    reference_tensor = PT(
        child=np.ones_like(reference_data),
        data_subject=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )
    gamma_tensor = reference_tensor.gamma

    output = gamma_tensor < gamma_tensor.ones_like() + 5
    assert output.shape == reference_tensor.shape
    assert output.child.all()
    assert list(output.sources.keys()) == [reference_tensor.id]
    state = {}
    for key in output.sources:
        state[key] = output.sources[key].child
    assert (output.func(state) == output.child).all()


@pytest.mark.equality
@pytest.mark.private_op
def test_gt_private(
    reference_data: np.ndarray,
    upper_bound: lra,
    lower_bound: lra,
    ishan: DataSubject,
) -> None:
    reference_tensor = PT(
        child=np.ones_like(reference_data),
        data_subject=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )
    gamma_tensor = reference_tensor.gamma

    output = gamma_tensor + 5 > gamma_tensor.zeros_like()
    assert output.shape == reference_tensor.shape
    assert list(output.sources.keys()) == [reference_tensor.id]
    state = {}
    for key in output.sources:
        state[key] = output.sources[key].child
    assert (output.func(state) == output.child).all()


@pytest.mark.equality
@pytest.mark.private_op
def test_le_private(
    reference_data: np.ndarray,
    upper_bound: lra,
    lower_bound: lra,
    ishan: DataSubject,
) -> None:
    ishan = np.broadcast_to(ishan, reference_data.shape)
    reference_tensor = PT(
        child=np.ones_like(reference_data),
        data_subject=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )
    gamma_tensor = reference_tensor.gamma

    output = gamma_tensor <= gamma_tensor.ones_like() + 5
    assert output.shape == reference_tensor.shape
    assert output.child.all()
    assert list(output.sources.keys()) == [reference_tensor.id]
    state = {}
    for key in output.sources:
        state[key] = output.sources[key].child
    assert (output.func(state) == output.child).all()


@pytest.mark.equality
@pytest.mark.private_op
def test_ge_private(
    reference_data: np.ndarray,
    upper_bound: lra,
    lower_bound: lra,
    ishan: DataSubject,
) -> None:
    ishan = np.broadcast_to(ishan, reference_data.shape)
    reference_tensor = PT(
        child=np.ones_like(reference_data),
        data_subject=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )
    gamma_tensor = reference_tensor.gamma

    output = gamma_tensor + 5 >= gamma_tensor.zeros_like()
    assert output.shape == reference_tensor.shape
    assert output.child.all()
    assert list(output.sources.keys()) == [reference_tensor.id]
    state = {}
    for key in output.sources:
        state[key] = output.sources[key].child
    assert (output.func(state) == output.child).all()


def test_resize(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubject,
) -> None:

    reference_tensor = PT(
        child=reference_data,
        data_subject=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )
    gamma_tensor = reference_tensor.gamma

    new_shape = tuple(map(lambda x: x * 2, reference_data.shape))
    output = gamma_tensor.resize(new_shape)
    ref_data = np.resize(reference_data, new_shape)
    flatten_ref = ref_data.flatten()
    flatten_res = output.child.flatten()

    assert (flatten_ref == flatten_res).all()
    assert list(output.sources.keys()) == [reference_tensor.id]
    state = {}
    for key in output.sources:
        state[key] = output.sources[key].child
    assert (output.func(state) == output.child).all()


def test_floordiv(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubject,
    dims: int,
) -> None:
    tensor = PT(
        child=reference_data,
        data_subject=ishan,
        min_vals=lower_bound,
        max_vals=upper_bound,
    )
    gamma_tensor = tensor.gamma
    # output = gamma_tensor // 5
    # assert (output.child == (reference_data // 5)).all()
    # assert list(output.sources.keys()) == [tensor.id]
    # state = {}
    # for key in output.sources:
    #     state[key] = output.sources[key].child
    # assert (output.func(state) == output.child).all()

    tensor2 = PT(
        child=reference_data,
        data_subject=ishan,
        min_vals=lower_bound,
        max_vals=upper_bound,
    )

    output = gamma_tensor // tensor2
    assert (output.child == jnp.floor_divide(reference_data, reference_data)).all()
    assert list(output.sources.keys()) == [tensor.id, tensor2.id]
    state = {}
    for key in output.sources:
        state[key] = output.sources[key].child
    print(output.child)
    print(output.func(state))
    assert (output.func(state) == output.child).all()

    array = np.ones((dims, dims))

    output = gamma_tensor // array
    assert (output.child == (reference_data // array)).all()
    assert list(output.sources.keys()) == [tensor.id]
    state = {}
    for key in output.sources:
        state[key] = output.sources[key].child
    assert (output.func(state) == output.child).all()


def test_prod(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubject,
    dims: int,
) -> None:
    tensor = PT(
        child=reference_data,
        data_subject=ishan,
        min_vals=lower_bound,
        max_vals=upper_bound,
    )
    gamma_tensor = tensor.gamma
    output = gamma_tensor.prod()
    assert (output.child == (reference_data.prod())).all()
    output = gamma_tensor.prod(axis=1)
    assert (output.child == (reference_data.prod(axis=1))).all()

    assert list(output.sources.keys()) == [tensor.id]
    state = {}
    for key in output.sources:
        state[key] = output.sources[key].child
    assert (output.func(state) == output.child).all()


def test_compress(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubject,
) -> None:
    reference_tensor = PT(
        child=reference_data,
        data_subject=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )
    gamma_tensor = reference_tensor.gamma

    condition = np.random.choice(a=[False, True], size=(reference_data.shape[0]))
    # if we have all False compress throws an exception because the size of the slices is 0
    while not any(condition):
        condition = np.random.choice(a=[False, True], size=(reference_data.shape[0]))
    
    print(condition.ndim)
    compressed_tensor = gamma_tensor.compress(condition, axis=0)

    comp_ind = 0
    for i, cond in enumerate(condition):
        if cond:
            assert (
                compressed_tensor.child[comp_ind, :] == reference_tensor.child[i, :]
            ).all()
            comp_ind += 1

    output = compressed_tensor
    assert list(output.sources.keys()) == [reference_tensor.id]
    state = {}
    for key in output.sources:
        state[key] = output.sources[key].child
    assert (output.func(state) == output.child).all()


def test_squeeze(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubject,
) -> None:
    new_reference_data = np.expand_dims(reference_data, axis=0)
    reference_tensor = PT(
        child=new_reference_data,
        data_subject=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )
    gamma_tensor = reference_tensor.gamma

    squeezed_tensor = gamma_tensor.squeeze()
    assert reference_tensor == squeezed_tensor.sources[reference_tensor.id]
    assert squeezed_tensor.shape == reference_data.shape
    output = squeezed_tensor
    assert list(output.sources.keys()) == [reference_tensor.id]
    state = {}
    for key in output.sources:
        state[key] = output.sources[key].child
    assert (output.func(state) == output.child).all()


def test_pos(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubject,
) -> None:
    reference_tensor = PT(
        child=reference_data,
        data_subject=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )
    gamma_tensor = reference_tensor.gamma

    output = +gamma_tensor

    assert reference_tensor == output.sources[reference_tensor.id]
    assert (output.child == reference_tensor.child).all()
    state = {}
    for key in output.sources:
        state[key] = output.sources[key].child
    assert (output.func(state) == output.child).all()


def test_neg(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubject,
) -> None:
    """Test neg for PT"""
    reference_tensor = PT(
        child=reference_data,
        data_subject=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )
    gamma_tensor = reference_tensor.gamma

    neg_tensor = gamma_tensor.__neg__()

    assert reference_tensor == neg_tensor.sources[reference_tensor.id]
    assert (neg_tensor.child == reference_tensor.child * -1).all()
    output = neg_tensor
    assert list(output.sources.keys()) == [reference_tensor.id]
    state = {}
    for key in output.sources:
        state[key] = output.sources[key].child
    assert (output.func(state) == output.child).all()


def test_any(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubject,
) -> None:
    reference_tensor = PT(
        child=np.array(reference_data),
        data_subject=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )
    gamma_tensor = reference_tensor.gamma

    aux_tensor = gamma_tensor == reference_data
    result = aux_tensor.any()
    assert result.child
    assert list(result.sources.keys()) == [reference_tensor.id]
    state = {}
    for key in result.sources:
        state[key] = result.sources[key].child
    assert (result.func(state) == result.child).all()

    result = (gamma_tensor == reference_data).any(axis=0)
    assert result.shape == (reference_data.shape[0],)
    assert list(result.sources.keys()) == [reference_tensor.id]
    state = {}
    for key in result.sources:
        state[key] = result.sources[key].child
    assert (result.func(state) == result.child).all()

    result = (gamma_tensor == reference_data).any(keepdims=True)
    assert result.shape == (1, 1)
    assert list(result.sources.keys()) == [reference_tensor.id]
    state = {}
    for key in result.sources:
        state[key] = result.sources[key].child
    assert (result.func(state) == result.child).all()

    result = (gamma_tensor == reference_data).any(keepdims=True, axis=0)
    assert result.shape == (1, reference_tensor.shape[0])
    assert list(result.sources.keys()) == [reference_tensor.id]
    state = {}
    for key in result.sources:
        state[key] = result.sources[key].child
    assert (result.func(state) == result.child).all()

def test_all(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubject,
) -> None:
    reference_tensor = PT(
        child=reference_data,
        data_subject=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )
    gamma_tensor = reference_tensor.gamma

    aux_tensor = gamma_tensor == reference_data
    result = aux_tensor.all()
    assert result.child
    assert list(result.sources.keys()) == [reference_tensor.id]
    state = {}
    for key in result.sources:
        state[key] = result.sources[key].child
    assert (result.func(state) == result.child).all()

    result = (gamma_tensor == reference_data).all(axis=0)
    assert result.shape == (reference_data.shape[0],)
    assert list(result.sources.keys()) == [reference_tensor.id]
    state = {}
    for key in result.sources:
        state[key] = result.sources[key].child
    assert (result.func(state) == result.child).all()

    result = (gamma_tensor == reference_data).all(keepdims=True)
    assert result.shape == (1, 1)
    assert list(result.sources.keys()) == [reference_tensor.id]
    state = {}
    for key in result.sources:
        state[key] = result.sources[key].child
    assert (result.func(state) == result.child).all()

    result = (gamma_tensor == reference_data).all(keepdims=True, axis=0)
    assert result.shape == (1, reference_tensor.shape[0])
    assert list(result.sources.keys()) == [reference_tensor.id]
    state = {}
    for key in result.sources:
        state[key] = result.sources[key].child
    assert (result.func(state) == result.child).all()


def test_copy(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubject,
) -> None:
    """Test copy for PT"""
    reference_tensor = PT(
        child=reference_data,
        data_subject=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )
    gamma_tensor = reference_tensor.gamma

    # Copy the tensor and check if it works
    copy_tensor = gamma_tensor.copy(order="K")  # jax implemented only 'K' order

    assert reference_tensor == copy_tensor.sources[reference_tensor.id]
    assert (
        reference_tensor.child == copy_tensor.child
    ).all(), "Copying of the PT fails"
    assert list(copy_tensor.sources.keys()) == [reference_tensor.id]
    state = {}
    for key in copy_tensor.sources:
        state[key] = copy_tensor.sources[key].child
    assert (copy_tensor.func(state) == copy_tensor.child).all()


def test_take(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubject,
) -> None:
    reference_tensor = PT(
        child=reference_data,
        data_subject=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )
    gamma_tensor = reference_tensor.gamma

    indices = np.array([2])
    result = gamma_tensor.take(indices, axis=0)
    assert reference_tensor == result.sources[reference_tensor.id]
    assert (result.child == reference_tensor.child[indices, :]).all()
    assert list(result.sources.keys()) == [reference_tensor.id]
    state = {}
    for key in result.sources:
        state[key] = result.sources[key].child
    assert (result.func(state) == result.child).all()


# Currently jax is not supporting put
# def test_put(
#     reference_data: np.ndarray,
#     upper_bound: np.ndarray,
#     lower_bound: np.ndarray,
#     ishan: DataSubject,
# ) -> None:
#     reference_tensor = PT(
#         child=reference_data,
#         data_subject=ishan,
#         max_vals=upper_bound,
#         min_vals=lower_bound,
#     )
#     gamma_tensor = reference_tensor.gamma

#     no_values = reference_tensor.shape[0]
#     new_values = np.random.randint(low=-5, high=5, size=(no_values), dtype=np.int32)
#     indices = np.random.randint(
#         low=0, high=no_values * no_values - no_values - 1, size=(1), dtype=np.int32
#     )[0]

#     result = gamma_tensor.put(range(indices, indices + no_values), new_values)
# assert reference_tensor == result.sources[reference_tensor.id]
# assert (
#     result.child.flat[indices : indices + no_values] == new_values  # noqa: E203
# ).all()
# assert list(result.sources.keys()) == [reference_tensor.id]
# state = {}
# for key in result.sources:
#     state[key] = result.sources[key].child
# assert (result.func(state) == result.child).all()


def test_abs(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubject,
) -> None:
    reference_tensor = PT(
        child=reference_data,
        data_subject=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )

    gamma_tensor = reference_tensor.gamma

    result = abs(gamma_tensor)
    assert reference_tensor == result.sources[reference_tensor.id]
    assert (result.child == abs(reference_tensor.child)).all()

    assert list(result.sources.keys()) == [reference_tensor.id]
    state = {}
    for key in result.sources:
        state[key] = result.sources[key].child
    assert (result.func(state) == result.child).all()


def test_argmax(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubject,
) -> None:
    reference_tensor = PT(
        child=reference_data,
        data_subject=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )
    gamma_tensor = reference_tensor.gamma

    result = gamma_tensor.argmax()
    reference_result = reference_tensor.child.argmax()
    assert reference_tensor == result.sources[reference_tensor.id]
    assert (result.child == reference_result).all()
    assert list(result.sources.keys()) == [reference_tensor.id]
    state = {}
    for key in result.sources:
        state[key] = result.sources[key].child
    assert (result.func(state) == result.child).all()

    result = gamma_tensor.argmax(axis=0)
    reference_result = reference_tensor.child.argmax(axis=0)
    assert reference_tensor == result.sources[reference_tensor.id]
    assert (result.child == reference_result).all()
    assert list(result.sources.keys()) == [reference_tensor.id]
    state = {}
    for key in result.sources:
        state[key] = result.sources[key].child
    assert (result.func(state) == result.child).all()


def test_argmin(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubject,
) -> None:
    reference_tensor = PT(
        child=reference_data,
        data_subject=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )
    gamma_tensor = reference_tensor.gamma

    result = gamma_tensor.argmin()
    reference_result = reference_tensor.child.argmin()
    assert reference_tensor == result.sources[reference_tensor.id]
    assert (result.child == reference_result).all()
    assert list(result.sources.keys()) == [reference_tensor.id]
    state = {}
    for key in result.sources:
        state[key] = result.sources[key].child
    assert (result.func(state) == result.child).all()

    result = gamma_tensor.argmin(axis=0)
    reference_result = reference_tensor.child.argmin(axis=0)
    assert reference_tensor == result.sources[reference_tensor.id]
    assert (result.child == reference_result).all()
    assert list(result.sources.keys()) == [reference_tensor.id]
    state = {}
    for key in result.sources:
        state[key] = result.sources[key].child
    assert (result.func(state) == result.child).all()


def test_swapaxes(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubject,
) -> None:
    reference_tensor = PT(
        child=reference_data,
        data_subject=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )
    gamma_tensor = reference_tensor.gamma

    result = gamma_tensor.swapaxes(0, 1)
    reference_result = reference_data.swapaxes(0, 1)
    assert reference_tensor == result.sources[reference_tensor.id]
    assert (result.child == reference_result).all()
    assert list(result.sources.keys()) == [reference_tensor.id]
    state = {}
    for key in result.sources:
        state[key] = result.sources[key].child
    assert (result.func(state) == result.child).all()


def test_ptp(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubject,
) -> None:
    reference_tensor = PT(
        child=reference_data,
        data_subject=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )
    gamma_tensor = reference_tensor.gamma

    result = gamma_tensor.ptp()
    assert reference_tensor == result.sources[reference_tensor.id]
    assert result.child == reference_data.ptp()
    assert list(result.sources.keys()) == [reference_tensor.id]
    state = {}
    for key in result.sources:
        state[key] = result.sources[key].child
    assert (result.func(state) == result.child).all()

    result = gamma_tensor.ptp(axis=0)
    assert (result.child == reference_data.ptp(axis=0, keepdims=True)).all()
    assert list(result.sources.keys()) == [reference_tensor.id]
    state = {}
    for key in result.sources:
        state[key] = result.sources[key].child
    assert (result.func(state) == result.child).all()


def test_nonzero(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubject,
) -> None:
    reference_tensor = PT(
        child=reference_data,
        data_subject=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )
    gamma_tensor = reference_tensor.gamma

    result = gamma_tensor.nonzero()
    reference_result = np.array(reference_data.nonzero())
    assert reference_tensor == result.sources[reference_tensor.id]
    assert (result.child == reference_result).all()
    assert list(result.sources.keys()) == [reference_tensor.id]
    state = {}
    for key in result.sources:
        state[key] = result.sources[key].child
    assert (result.func(state)[0] == result.child[0]).all()
    assert (result.func(state)[1] == result.child[1]).all()


def test_var(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubject,
) -> None:
    tensor = PT(
        child=reference_data,
        data_subject=ishan,
        min_vals=lower_bound,
        max_vals=upper_bound,
    )
    gamma_tensor = tensor.gamma

    result = gamma_tensor.var()
    assert result.child == jnp.var(reference_data)
    assert list(result.sources.keys()) == [tensor.id]
    state = {}
    for key in result.sources:
        state[key] = result.sources[key].child
    assert (result.func(state) == result.child).all()

    result = gamma_tensor.var(axis=1)
    assert (result.child == jnp.var(reference_data, axis=1)).all()
    assert list(result.sources.keys()) == [tensor.id]
    state = {}
    for key in result.sources:
        state[key] = result.sources[key].child
    assert (result.func(state) == result.child).all()


def test_cumsum(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubject,
) -> None:
    tensor = PT(
        child=reference_data,
        data_subject=ishan,
        min_vals=lower_bound,
        max_vals=upper_bound,
    )
    gamma_tensor = tensor.gamma

    result = gamma_tensor.cumsum()
    assert (result.child == reference_data.cumsum()).all()
    assert list(result.sources.keys()) == [tensor.id]
    state = {}
    for key in result.sources:
        state[key] = result.sources[key].child
    assert (result.func(state) == result.child).all()

    result = gamma_tensor.cumsum(axis=1)
    assert (result.child == reference_data.cumsum(axis=1)).all()
    assert list(result.sources.keys()) == [tensor.id]
    state = {}
    for key in result.sources:
        state[key] = result.sources[key].child
    assert (result.func(state) == result.child).all()


def test_std(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubject,
) -> None:
    tensor = PT(
        child=reference_data,
        data_subject=ishan,
        min_vals=lower_bound,
        max_vals=upper_bound,
    )
    gamma_tensor = tensor.gamma

    result = gamma_tensor.std()
    assert result.child == jnp.std(reference_data)
    assert list(result.sources.keys()) == [tensor.id]
    state = {}
    for key in result.sources:
        state[key] = result.sources[key].child
    assert (result.func(state) == result.child).all()

    result = gamma_tensor.std(axis=1)
    assert (result.child == jnp.std(reference_data, axis=1)).all()
    assert list(result.sources.keys()) == [tensor.id]
    state = {}
    for key in result.sources:
        state[key] = result.sources[key].child
    assert (result.func(state) == result.child).all()


def test_trace(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubject,
) -> None:
    tensor = PT(
        child=reference_data,
        data_subject=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )
    gamma_tensor = tensor.gamma

    result = gamma_tensor.trace()
    assert result.child == reference_data.trace()
    assert list(result.sources.keys()) == [tensor.id]
    state = {}
    for key in result.sources:
        state[key] = result.sources[key].child
    assert (result.func(state) == result.child).all()

    result = gamma_tensor.trace(offset=1)
    assert result.child == reference_data.trace(offset=1)
    assert list(result.sources.keys()) == [tensor.id]
    state = {}
    for key in result.sources:
        state[key] = result.sources[key].child
    assert (result.func(state) == result.child).all()


def test_cumprod(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubject,
) -> None:
    # Note: It's difficult to test the min/max values for cumprod because of the extremely high bounds this op gives.
    tensor = PT(
        child=reference_data,
        data_subject=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )
    gamma_tensor = tensor.gamma

    result = gamma_tensor.cumprod()
    assert (result.child == reference_data.cumprod()).all()
    assert list(result.sources.keys()) == [tensor.id]
    state = {}
    for key in result.sources:
        state[key] = result.sources[key].child
    assert (result.func(state) == result.child).all()

    result = gamma_tensor.cumprod(axis=1)
    assert (result.child == reference_data.cumprod(axis=1)).all()
    assert list(result.sources.keys()) == [tensor.id]
    state = {}
    for key in result.sources:
        state[key] = result.sources[key].child
    assert (result.func(state) == result.child).all()


def test_max(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubject,
) -> None:
    tensor = PT(
        child=reference_data,
        data_subject=ishan,
        min_vals=lower_bound,
        max_vals=upper_bound,
    )
    gamma_tensor = tensor.gamma

    result = gamma_tensor.max()
    assert result.child == reference_data.max()
    assert list(result.sources.keys()) == [tensor.id]
    state = {}
    for key in result.sources:
        state[key] = result.sources[key].child
    assert (result.func(state) == result.child).all()


def test_min(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubject,
) -> None:
    tensor = PT(
        child=reference_data,
        data_subject=ishan,
        min_vals=lower_bound,
        max_vals=upper_bound,
    )
    gamma_tensor = tensor.gamma

    result = gamma_tensor.min()
    assert result.child == reference_data.min()
    assert list(result.sources.keys()) == [tensor.id]
    state = {}
    for key in result.sources:
        state[key] = result.sources[key].child
    assert (result.func(state) == result.child).all()


def test_matmul(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubject,
) -> None:
    reference_tensor = PT(
        child=reference_data,
        data_subject=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )
    gamma_tensor = reference_tensor.gamma

    result = gamma_tensor @ gamma_tensor
    assert (result.child == (reference_data @ reference_data)).all()
    assert list(result.sources.keys()) == [reference_tensor.id]
    state = {}
    for key in result.sources:
        state[key] = result.sources[key].child
    assert (result.func(state) == result.child).all()


def test_lshift(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubject,
) -> None:
    tensor = PT(
        child=reference_data,
        data_subject=ishan,
        min_vals=lower_bound,
        max_vals=upper_bound,
    )
    gamma_tensor = tensor.gamma

    result = gamma_tensor << 10
    assert (result.child == reference_data << 10).all()
    assert list(result.sources.keys()) == [tensor.id]
    state = {}
    for key in result.sources:
        state[key] = result.sources[key].child
    assert (result.func(state) == result.child).all()


def test_rshift(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubject,
) -> None:
    tensor = PT(
        child=reference_data,
        data_subject=ishan,
        min_vals=lower_bound,
        max_vals=upper_bound,
    )
    gamma_tensor = tensor.gamma

    result = gamma_tensor >> 10
    assert (result.child == reference_data >> 10).all()
    assert list(result.sources.keys()) == [tensor.id]
    state = {}
    for key in result.sources:
        state[key] = result.sources[key].child
    assert (result.func(state) == result.child).all()


def test_round(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubject,
) -> None:
    data = reference_data / 100
    tensor = PT(
        child=data,
        data_subject=ishan,
        min_vals=lower_bound / 100,
        max_vals=upper_bound / 100,
    )
    gamma_tensor = tensor.gamma

    result = gamma_tensor.round()
    assert (result.child == data.round()).all()
    assert list(result.sources.keys()) == [tensor.id]
    state = {}
    for key in result.sources:
        state[key] = result.sources[key].child
    assert (result.func(state) == result.child).all()

    result = gamma_tensor.round(2)
    assert (result.child == data.round(2)).all()
    assert list(result.sources.keys()) == [tensor.id]
    state = {}
    for key in result.sources:
        state[key] = result.sources[key].child
    assert (result.func(state) == result.child).all()


def test_sort(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubject,
) -> None:
    tensor = PT(
        child=reference_data,
        data_subject=ishan,
        min_vals=lower_bound,
        max_vals=upper_bound,
    )
    gamma_tensor = tensor.gamma

    result = gamma_tensor.sort()
    data = reference_data.copy()  # make a copy incase this changes the fixture
    data.sort()
    assert (result.child == data).all()
    assert list(result.sources.keys()) == [tensor.id]
    state = {}
    for key in result.sources:
        state[key] = result.sources[key].child
    assert (result.func(state) == result.child).all()

    result = gamma_tensor.sort(axis=1)
    data = reference_data.copy()  # make a copy incase this changes the fixture
    data.sort(axis=1)
    assert (result.child == data).all()
    assert list(result.sources.keys()) == [tensor.id]
    state = {}
    for key in result.sources:
        state[key] = result.sources[key].child
    assert (result.func(state) == result.child).all()


def test_argsort(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubject,
) -> None:
    tensor = PT(
        child=reference_data,
        data_subject=ishan,
        min_vals=lower_bound,
        max_vals=upper_bound,
    )
    gamma_tensor = tensor.gamma
    result = gamma_tensor.argsort()
    assert (result.child == reference_data.argsort()).all()
    assert list(result.sources.keys()) == [tensor.id]
    state = {}
    for key in result.sources:
        state[key] = result.sources[key].child
    assert (result.func(state) == result.child).all()

    result = gamma_tensor.argsort(axis=1)
    assert (result.child == reference_data.argsort(axis=1)).all()
    assert list(result.sources.keys()) == [tensor.id]
    state = {}
    for key in result.sources:
        state[key] = result.sources[key].child
    assert (result.func(state) == result.child).all()


def test_transpose(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubject,
) -> None:
    tensor = PT(
        child=reference_data,
        data_subject=ishan,
        min_vals=lower_bound,
        max_vals=upper_bound,
    )
    gamma_tensor = tensor.gamma
    result = gamma_tensor.transpose()
    assert (result.child == reference_data.transpose()).all()
    assert list(result.sources.keys()) == [tensor.id]
    state = {}
    for key in result.sources:
        state[key] = result.sources[key].child
    assert (result.func(state) == result.child).all()

    result = gamma_tensor.transpose((1, 0))
    assert (result.child == reference_data.transpose((1, 0))).all()
    assert list(result.sources.keys()) == [tensor.id]
    state = {}
    for key in result.sources:
        state[key] = result.sources[key].child
    assert (result.func(state) == result.child).all()


def test_reshape(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubject,
    dims: int,
) -> None:
    tensor = PT(
        child=reference_data,
        data_subject=ishan,
        min_vals=lower_bound,
        max_vals=upper_bound,
    )
    gamma_tensor = tensor.gamma

    result = gamma_tensor.reshape((1, dims * dims))
    assert (result.child == reference_data.reshape((1, dims * dims))).all()
    assert list(result.sources.keys()) == [tensor.id]
    state = {}
    for key in result.sources:
        state[key] = result.sources[key].child
    assert (result.func(state) == result.child).all()


def test_xor(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubject,
) -> None:
    reference_tensor = PT(
        child=reference_data,
        data_subject=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )
    gamma_tensor = reference_tensor.gamma
    other = np.ones_like(reference_data)
    result = gamma_tensor ^ other

    assert (result.child == (reference_data ^ other)).all()
    assert list(result.sources.keys()) == [reference_tensor.id]
    state = {}
    for key in result.sources:
        state[key] = result.sources[key].child
    assert (result.func(state) == result.child).all()

    other = np.zeros_like(reference_data)
    result = gamma_tensor ^ other
    assert (result.child == (reference_data ^ other)).all()
    assert list(result.sources.keys()) == [reference_tensor.id]
    state = {}
    for key in result.sources:
        state[key] = result.sources[key].child
    assert (result.func(state) == result.child).all()

    result = gamma_tensor ^ reference_tensor
    assert (result.child == (reference_data ^ reference_data)).all()
    assert list(result.sources.keys()) == [reference_tensor.id]
    state = {}
    for key in result.sources:
        state[key] = result.sources[key].child
    assert (result.func(state) == result.child).all()

    other = PT(
        child=reference_data,
        data_subject=ishan,
        min_vals=lower_bound,
        max_vals=upper_bound,
    )
    gamma_tensor2 = other.gamma

    result = gamma_tensor ^ gamma_tensor2
    assert (result.child == (reference_data ^ other.child)).all()
    assert list(result.sources.keys()) == [reference_tensor.id, other.id]
    state = {}
    for key in result.sources:
        state[key] = result.sources[key].child
    assert (result.func(state) == result.child).all()
