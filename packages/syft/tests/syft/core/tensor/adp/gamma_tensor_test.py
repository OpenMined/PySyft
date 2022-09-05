# stdlib
from typing import Any

# third party
import numpy as np
from numpy.typing import ArrayLike
import pytest

# syft absolute
import syft as sy
from syft.core.adp.data_subject_ledger import DataSubjectLedger
from syft.core.adp.data_subject_list import DataSubjectArray
from syft.core.adp.ledger_store import DictLedgerStore
from syft.core.tensor.autodp.gamma_tensor import GammaTensor
from syft.core.tensor.autodp.phi_tensor import PhiTensor as PT
from syft.core.tensor.lazy_repeat_array import lazyrepeatarray as lra


@pytest.fixture
def ishan() -> ArrayLike:
    return np.array(DataSubjectArray(["φhishan"]))


@pytest.fixture
def traskmaster() -> ArrayLike:
    return np.ndarray(DataSubjectArray(["λamdrew"]))


@pytest.fixture
def highest() -> int:
    return 50


@pytest.fixture
def lowest(highest) -> int:
    return -1 * int(highest)


@pytest.fixture
def dims() -> int:
    """This generates a random integer for the number of dimensions in our testing tensors"""
    dims = int(max(3, np.random.randint(10) + 3))  # Avoid size 0 and 1
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
    data_subjects = np.broadcast_to(
        np.array(DataSubjectArray(["eagle"])), reference_data.shape
    )
    tensor1 = PT(
        child=reference_data,
        data_subjects=data_subjects,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )
    assert tensor1.data_subjects.shape == tensor1.child.shape
    gamma_tensor1 = tensor1.gamma

    print("gamma tensor", gamma_tensor1)
    # Checks to ensure gamma tensor was properly created
    assert isinstance(gamma_tensor1, GammaTensor)
    assert (gamma_tensor1.child == tensor1.child).all()

    ser = sy.serialize(gamma_tensor1, to_bytes=True)
    de = sy.deserialize(ser, from_bytes=True)

    assert (de.child == gamma_tensor1.child).all()
    assert (de.data_subjects == gamma_tensor1.data_subjects).all()
    assert de.min_vals == gamma_tensor1.min_vals
    assert de.max_vals == gamma_tensor1.max_vals
    assert de.is_linear == gamma_tensor1.is_linear
    assert de.func_str == gamma_tensor1.func_str
    assert de.id == gamma_tensor1.id
    assert de.sources.keys() == gamma_tensor1.sources.keys()


def test_gamma_publish(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
) -> None:
    """Test basic serde for GammaTensor"""
    data_subjects = np.broadcast_to(
        np.array(DataSubjectArray(["eagle", "potato"])), reference_data.shape
    )
    tensor1 = GammaTensor(
        child=reference_data,
        data_subjects=data_subjects,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )
    assert tensor1.data_subjects.shape == tensor1.child.shape
    gamma_tensor1 = tensor1.sum()
    assert isinstance(gamma_tensor1, GammaTensor)
    # Gamma Tensor Does not have FPT Values
    assert tensor1.child.sum() == gamma_tensor1.child

    ledger_store = DictLedgerStore()
    print(ledger_store.kv_store)
    user_key = b"1231"
    ledger = DataSubjectLedger.get_or_create(store=ledger_store, user_key=user_key)

    def get_budget_for_user(*args: Any, **kwargs: Any) -> float:
        return 999999

    def deduct_epsilon_for_user(*args: Any, **kwargs: Any) -> bool:
        return True

    results = gamma_tensor1.publish(
        get_budget_for_user=get_budget_for_user,
        deduct_epsilon_for_user=deduct_epsilon_for_user,
        ledger=ledger,
        sigma=0.1,
    )

    assert results.dtype == np.float64
    assert results < upper_bound.to_numpy().sum() + 10
    assert -10 + lower_bound.to_numpy().sum() < results
    print(ledger_store.kv_store)


@pytest.mark.arithmetic
@pytest.mark.public_op
def test_add_public(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubjectArray,
) -> None:
    ishan = np.broadcast_to(ishan, reference_data.shape)
    reference_tensor = PT(
        child=reference_data,
        data_subjects=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    ).gamma

    output = reference_tensor + 5
    assert output.shape == reference_tensor.shape
    assert (output.child == reference_data + 5).all()
    assert output.min_vals.data == reference_tensor.min_vals + 5
    assert output.min_vals.shape == reference_tensor.shape
    assert output.max_vals.data == reference_tensor.max_vals + 5
    assert output.max_vals.shape == reference_tensor.shape
    assert (output.data_subjects == reference_tensor.data_subjects).all()


@pytest.mark.arithmetic
@pytest.mark.public_op
def test_sub_public(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubjectArray,
) -> None:
    ishan = np.broadcast_to(ishan, reference_data.shape)
    reference_tensor = PT(
        child=reference_data,
        data_subjects=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    ).gamma

    output = reference_tensor - 5
    assert output.shape == reference_tensor.shape
    assert (output.child == reference_data - 5).all()
    assert output.min_vals.data == reference_tensor.min_vals - 5
    assert output.min_vals.shape == reference_tensor.shape
    assert output.max_vals.data == reference_tensor.max_vals - 5
    assert output.max_vals.shape == reference_tensor.shape
    assert (output.data_subjects == reference_tensor.data_subjects).all()


@pytest.mark.arithmetic
@pytest.mark.public_op
def test_mul_public(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubjectArray,
) -> None:
    ishan = np.broadcast_to(ishan, reference_data.shape)
    reference_tensor = PT(
        child=reference_data,
        data_subjects=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    ).gamma

    output = reference_tensor * 5
    assert output.shape == reference_tensor.shape
    assert (output.child == reference_data * 5).all()
    assert (output.min_vals.data == reference_tensor.min_vals.data * 5).all()
    assert output.min_vals.shape == reference_tensor.shape
    assert (output.max_vals.data == reference_tensor.max_vals.data * 5).all()
    assert output.max_vals.shape == reference_tensor.shape
    assert (output.data_subjects == reference_tensor.data_subjects).all()


@pytest.mark.arithmetic
@pytest.mark.public_op
def test_truediv_public(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubjectArray,
) -> None:
    ishan = np.broadcast_to(ishan, reference_data.shape)
    reference_tensor = PT(
        child=reference_data,
        data_subjects=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    ).gamma

    output = reference_tensor / 5
    assert output.shape == reference_tensor.shape
    assert (output.child == reference_data / 5).all()
    assert (output.min_vals.data == reference_tensor.min_vals.data / 5).all()
    assert output.min_vals.shape == reference_tensor.shape
    assert (output.max_vals.data == reference_tensor.max_vals.data / 5).all()
    assert output.max_vals.shape == reference_tensor.shape
    assert (output.data_subjects == reference_tensor.data_subjects).all()


@pytest.mark.arithmetic
@pytest.mark.private_op
def test_add_private(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubjectArray,
) -> None:
    ishan = np.broadcast_to(ishan, reference_data.shape)
    reference_tensor = PT(
        child=reference_data,
        data_subjects=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    ).gamma

    tensor2 = PT(
        child=reference_data,
        data_subjects=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    ).gamma

    output = reference_tensor + tensor2
    assert output.shape == reference_tensor.shape
    assert (output.child == reference_data * 2).all()
    assert output.min_vals.data == reference_tensor.min_vals.data * 2
    assert output.min_vals.shape == reference_tensor.shape
    assert output.max_vals.data == reference_tensor.max_vals.data * 2
    assert output.max_vals.shape == reference_tensor.shape
    assert (output.data_subjects == reference_tensor.data_subjects).all()


@pytest.mark.arithmetic
@pytest.mark.private_op
def test_sub_private(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubjectArray,
) -> None:
    ishan = np.broadcast_to(ishan, reference_data.shape)
    reference_tensor = PT(
        child=reference_data,
        data_subjects=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    ).gamma

    tensor2 = PT(
        child=reference_data,
        data_subjects=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    ).gamma

    output = reference_tensor - tensor2
    assert output.shape == reference_tensor.shape
    assert (output.child == 0).all()
    assert output.min_vals.data <= output.max_vals.data
    assert output.min_vals.shape == reference_tensor.shape
    assert output.max_vals.shape == reference_tensor.shape
    assert (output.data_subjects == reference_tensor.data_subjects).all()


@pytest.mark.arithmetic
@pytest.mark.private_op
def test_mul_private(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubjectArray,
) -> None:
    ishan = np.broadcast_to(ishan, reference_data.shape)
    reference_tensor = PT(
        child=reference_data,
        data_subjects=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    ).gamma

    tensor2 = PT(
        child=reference_data,
        data_subjects=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    ).gamma

    output = reference_tensor * tensor2
    assert output.shape == reference_tensor.shape
    assert (output.child == reference_data**2).all()
    assert output.min_vals.data <= output.max_vals.data
    assert output.min_vals.shape == reference_tensor.shape
    assert output.max_vals.shape == reference_tensor.shape
    assert (output.data_subjects == reference_tensor.data_subjects).all()


@pytest.mark.arithmetic
@pytest.mark.private_op
def test_truediv_private(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubjectArray,
) -> None:
    ishan = np.broadcast_to(ishan, reference_data.shape)
    reference_tensor = PT(
        child=np.ones_like(reference_data),
        data_subjects=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    ).gamma

    tensor2 = PT(
        child=np.ones_like(reference_data),
        data_subjects=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    ).gamma

    output = reference_tensor / tensor2
    assert output.shape == reference_tensor.shape
    assert (output.child == 1).all()
    assert output.min_vals.data <= output.max_vals.data
    assert output.min_vals.shape == reference_tensor.shape
    assert output.max_vals.shape == reference_tensor.shape
    assert (output.data_subjects == reference_tensor.data_subjects).all()


@pytest.mark.equality
@pytest.mark.public_op
def test_eq_public(
    reference_data: np.ndarray,
    upper_bound: lra,
    lower_bound: lra,
    ishan: DataSubjectArray,
) -> None:
    ishan = np.broadcast_to(ishan, reference_data.shape)
    reference_tensor = PT(
        child=np.ones_like(reference_data),
        data_subjects=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    ).gamma

    # Test that it IS equal
    output = reference_tensor == 1
    assert output.shape == reference_tensor.shape
    assert output.child.all()
    assert (output.min_vals.data == 0).all()
    assert output.min_vals.shape == reference_tensor.shape
    assert (output.max_vals.data == 1).all()
    assert output.max_vals.shape == reference_tensor.shape
    assert (output.data_subjects == reference_tensor.data_subjects).all()


@pytest.mark.equality
@pytest.mark.public_op
def test_ne_public(
    reference_data: np.ndarray,
    upper_bound: lra,
    lower_bound: lra,
    ishan: DataSubjectArray,
) -> None:
    ishan = np.broadcast_to(ishan, reference_data.shape)
    reference_tensor = PT(
        child=np.ones_like(reference_data),
        data_subjects=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    ).gamma

    output = reference_tensor != 0
    assert output.shape == reference_tensor.shape
    assert output.child.all()
    assert (output.min_vals.data == 0).all()
    assert output.min_vals.shape == reference_tensor.shape
    assert (output.max_vals.data == 1).all()
    assert output.max_vals.shape == reference_tensor.shape
    assert (output.data_subjects == reference_tensor.data_subjects).all()


@pytest.mark.equality
@pytest.mark.public_op
def test_lt_public(
    reference_data: np.ndarray,
    upper_bound: lra,
    lower_bound: lra,
    ishan: DataSubjectArray,
) -> None:
    ishan = np.broadcast_to(ishan, reference_data.shape)
    reference_tensor = PT(
        child=np.ones_like(reference_data),
        data_subjects=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    ).gamma

    output = reference_tensor < 2
    assert output.shape == reference_tensor.shape
    assert output.child.all()
    assert (output.min_vals.data == 0).all()
    assert output.min_vals.shape == reference_tensor.shape
    assert (output.max_vals.data == 1).all()
    assert output.max_vals.shape == reference_tensor.shape
    assert (output.data_subjects == reference_tensor.data_subjects).all()


@pytest.mark.equality
@pytest.mark.public_op
def test_gt_public(
    reference_data: np.ndarray,
    upper_bound: lra,
    lower_bound: lra,
    ishan: DataSubjectArray,
) -> None:
    ishan = np.broadcast_to(ishan, reference_data.shape)
    reference_tensor = PT(
        child=np.ones_like(reference_data),
        data_subjects=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    ).gamma

    output = reference_tensor > 0
    assert output.shape == reference_tensor.shape
    assert output.child.all()
    assert (output.min_vals.data == 0).all()
    assert output.min_vals.shape == reference_tensor.shape
    assert (output.max_vals.data == 1).all()
    assert output.max_vals.shape == reference_tensor.shape
    assert (output.data_subjects == reference_tensor.data_subjects).all()


@pytest.mark.equality
@pytest.mark.public_op
def test_le_public(
    reference_data: np.ndarray,
    upper_bound: lra,
    lower_bound: lra,
    ishan: DataSubjectArray,
) -> None:
    ishan = np.broadcast_to(ishan, reference_data.shape)
    reference_tensor = PT(
        child=np.ones_like(reference_data),
        data_subjects=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    ).gamma

    output = reference_tensor <= 2
    assert output.shape == reference_tensor.shape
    assert output.child.all()
    assert (output.min_vals.data == 0).all()
    assert output.min_vals.shape == reference_tensor.shape
    assert (output.max_vals.data == 1).all()
    assert output.max_vals.shape == reference_tensor.shape
    assert (output.data_subjects == reference_tensor.data_subjects).all()


@pytest.mark.equality
@pytest.mark.public_op
def test_ge_public(
    reference_data: np.ndarray,
    upper_bound: lra,
    lower_bound: lra,
    ishan: DataSubjectArray,
) -> None:
    ishan = np.broadcast_to(ishan, reference_data.shape)
    reference_tensor = PT(
        child=np.ones_like(reference_data),
        data_subjects=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    ).gamma

    output = reference_tensor >= 0
    assert output.shape == reference_tensor.shape
    assert output.child.all()
    assert (output.min_vals.data == 0).all()
    assert output.min_vals.shape == reference_tensor.shape
    assert (output.max_vals.data == 1).all()
    assert output.max_vals.shape == reference_tensor.shape
    assert (output.data_subjects == reference_tensor.data_subjects).all()


@pytest.mark.equality
@pytest.mark.private_op
def test_eq_private(
    reference_data: np.ndarray,
    upper_bound: lra,
    lower_bound: lra,
    ishan: DataSubjectArray,
) -> None:
    ishan = np.broadcast_to(ishan, reference_data.shape)
    reference_tensor = PT(
        child=np.ones_like(reference_data),
        data_subjects=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    ).gamma

    # Test that it IS equal
    output = reference_tensor == reference_tensor.ones_like()
    assert output.shape == reference_tensor.shape
    assert output.child.all()
    assert (output.min_vals.data == 0).all()
    assert output.min_vals.shape == reference_tensor.shape
    assert (output.max_vals.data == 1).all()
    assert output.max_vals.shape == reference_tensor.shape
    assert (output.data_subjects == reference_tensor.data_subjects).all()


@pytest.mark.equality
@pytest.mark.private_op
def test_ne_private(
    reference_data: np.ndarray,
    upper_bound: lra,
    lower_bound: lra,
    ishan: DataSubjectArray,
) -> None:
    ishan = np.broadcast_to(ishan, reference_data.shape)
    reference_tensor = PT(
        child=np.ones_like(reference_data),
        data_subjects=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    ).gamma

    output = reference_tensor != reference_tensor.zeros_like()
    assert output.shape == reference_tensor.shape
    assert output.child.all()
    assert (output.min_vals.data == 0).all()
    assert output.min_vals.shape == reference_tensor.shape
    assert (output.max_vals.data == 1).all()
    assert output.max_vals.shape == reference_tensor.shape
    assert (output.data_subjects == reference_tensor.data_subjects).all()


@pytest.mark.equality
@pytest.mark.private_op
def test_lt_private(
    reference_data: np.ndarray,
    upper_bound: lra,
    lower_bound: lra,
    ishan: DataSubjectArray,
) -> None:
    ishan = np.broadcast_to(ishan, reference_data.shape)
    reference_tensor = PT(
        child=np.ones_like(reference_data),
        data_subjects=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    ).gamma

    output = reference_tensor < reference_tensor.ones_like() + 5
    assert output.shape == reference_tensor.shape
    assert output.child.all()
    assert (output.min_vals.data == 0).all()
    assert output.min_vals.shape == reference_tensor.shape
    assert (output.max_vals.data == 1).all()
    assert output.max_vals.shape == reference_tensor.shape
    assert (output.data_subjects == reference_tensor.data_subjects).all()


@pytest.mark.equality
@pytest.mark.private_op
def test_gt_private(
    reference_data: np.ndarray,
    upper_bound: lra,
    lower_bound: lra,
    ishan: DataSubjectArray,
) -> None:
    ishan = np.broadcast_to(ishan, reference_data.shape)
    reference_tensor = PT(
        child=np.ones_like(reference_data),
        data_subjects=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    ).gamma

    output = reference_tensor + 5 > reference_tensor.zeros_like()
    assert output.shape == reference_tensor.shape
    assert output.child.all()
    assert (output.min_vals.data == 0).all()
    assert output.min_vals.shape == reference_tensor.shape
    assert (output.max_vals.data == 1).all()
    assert output.max_vals.shape == reference_tensor.shape
    assert (output.data_subjects == reference_tensor.data_subjects).all()


@pytest.mark.equality
@pytest.mark.private_op
def test_le_private(
    reference_data: np.ndarray,
    upper_bound: lra,
    lower_bound: lra,
    ishan: DataSubjectArray,
) -> None:
    ishan = np.broadcast_to(ishan, reference_data.shape)
    reference_tensor = PT(
        child=np.ones_like(reference_data),
        data_subjects=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    ).gamma

    output = reference_tensor <= reference_tensor.ones_like() + 5
    assert output.shape == reference_tensor.shape
    assert output.child.all()
    assert (output.min_vals.data == 0).all()
    assert output.min_vals.shape == reference_tensor.shape
    assert (output.max_vals.data == 1).all()
    assert output.max_vals.shape == reference_tensor.shape
    assert (output.data_subjects == reference_tensor.data_subjects).all()


@pytest.mark.equality
@pytest.mark.private_op
def test_ge_private(
    reference_data: np.ndarray,
    upper_bound: lra,
    lower_bound: lra,
    ishan: DataSubjectArray,
) -> None:
    ishan = np.broadcast_to(ishan, reference_data.shape)
    reference_tensor = PT(
        child=np.ones_like(reference_data),
        data_subjects=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    ).gamma

    output = reference_tensor + 5 >= reference_tensor.zeros_like()
    assert output.shape == reference_tensor.shape
    assert output.child.all()
    assert (output.min_vals.data == 0).all()
    assert output.min_vals.shape == reference_tensor.shape
    assert (output.max_vals.data == 1).all()
    assert output.max_vals.shape == reference_tensor.shape
    assert (output.data_subjects == reference_tensor.data_subjects).all()


def test_resize(
    reference_data: np.ndarray,
    upper_bound: lra,
    lower_bound: lra,
    ishan: DataSubjectArray,
) -> None:
    ishan = np.broadcast_to(ishan, reference_data.shape)
    reference_tensor = PT(
        child=reference_data,
        data_subjects=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    ).gamma

    new_shape = tuple(map(lambda x: x * 2, reference_data.shape))
    resized_tensor = reference_tensor.resize(new_shape)

    no_of_elems = new_shape[0] * new_shape[1] // 4

    flatten_ref = reference_tensor.child.flatten()
    flatten_res = resized_tensor.child.flatten()

    assert resized_tensor.func_str == "resize"
    assert reference_tensor == resized_tensor.sources[reference_tensor.id]

    assert (flatten_ref == flatten_res[0:no_of_elems]).all()
    assert (flatten_ref == flatten_res[no_of_elems : no_of_elems * 2]).all()
    assert (flatten_ref == flatten_res[no_of_elems * 2 : no_of_elems * 3]).all()
    assert (flatten_ref == flatten_res[no_of_elems * 3 : no_of_elems * 4]).all()

    assert resized_tensor.min_vals.shape == new_shape
    assert resized_tensor.max_vals.shape == new_shape

    data_subjects_ref = reference_tensor.data_subjects.flatten()
    data_subjects_res = resized_tensor.data_subjects.flatten()
    assert (data_subjects_ref == data_subjects_res[0:no_of_elems]).all()
    assert (data_subjects_ref == data_subjects_res[no_of_elems : no_of_elems * 2]).all()
    assert (
        data_subjects_ref == data_subjects_res[no_of_elems * 2 : no_of_elems * 3]
    ).all()
    assert (
        data_subjects_ref == data_subjects_res[no_of_elems * 3 : no_of_elems * 4]
    ).all()


def test_compress(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubjectArray,
) -> None:
    ishan = np.broadcast_to(ishan, reference_data.shape)
    reference_tensor = PT(
        child=reference_data,
        data_subjects=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    ).gamma

    condition = list(np.random.choice(a=[False, True], size=(reference_data.shape[0])))
    compressed_tensor = reference_tensor.compress(condition, axis=0)

    assert compressed_tensor.func_str == "compress"
    assert reference_tensor == compressed_tensor.sources[reference_tensor.id]

    new_shape = (
        reference_tensor.shape[0] - len([0 for c in condition if not c]),
        reference_tensor.shape[1],
    )

    comp_ind = 0
    for i, cond in enumerate(condition):
        if cond:
            assert (
                compressed_tensor.child[comp_ind, :] == reference_tensor.child[i, :]
            ).all()
            assert (
                compressed_tensor.data_subjects[comp_ind, :]
                == reference_tensor.data_subjects[i, :]
            ).all()
            comp_ind += 1

    assert compressed_tensor.min_vals.shape == new_shape
    assert compressed_tensor.max_vals.shape == new_shape


def test_squeeze(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubjectArray,
) -> None:
    new_reference_data = np.expand_dims(reference_data, axis=0)
    ishan = np.broadcast_to(ishan, reference_data.shape)
    new_ishan = np.broadcast_to(ishan, new_reference_data.shape)
    reference_tensor = PT(
        child=new_reference_data,
        data_subjects=new_ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    ).gamma

    squeezed_tensor = reference_tensor.squeeze()
    assert squeezed_tensor.func_str == "squeeze"
    assert reference_tensor == squeezed_tensor.sources[reference_tensor.id]
    assert squeezed_tensor.shape == reference_data.shape
    assert (squeezed_tensor.child == reference_data).all()
    assert (squeezed_tensor.data_subjects == ishan).all()
