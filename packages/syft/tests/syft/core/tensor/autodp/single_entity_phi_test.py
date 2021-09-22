# stdlib
from random import randint
from random import sample

# third party
import numpy as np
import pytest

# syft absolute
from syft import deserialize
from syft import serialize
from syft.core.adp.entity import Entity
from syft.core.tensor.autodp.single_entity_phi import SingleEntityPhiTensor
from syft.core.tensor.tensor import Tensor

gonzalo = Entity(name="Gonzalo")


@pytest.fixture(scope="function")
def x() -> Tensor:
    x = Tensor(np.array([[1, 2, 3], [4, 5, 6]]))
    x = x.private(min_val=-1, max_val=7, entities=[gonzalo])
    return x


@pytest.fixture(scope="function")
def y() -> Tensor:
    y = Tensor(np.array([[-1, -2, -3], [-4, -5, -6]]))
    y = y.private(min_val=-7, max_val=1, entities=[gonzalo])
    return y


ent = Entity(name="test")
ent2 = Entity(name="test2")

dims = np.random.randint(10) + 1

child1 = np.random.randint(low=-2, high=4, size=dims)
upper1 = np.full(dims, 3, dtype=np.int32)
low1 = np.full(dims, -2, dtype=np.int32)
child2 = np.random.randint(low=4, high=7, size=dims)
upper2 = np.full(dims, 6, dtype=np.int32)
low2 = np.full(dims, 4, dtype=np.int32)

tensor1 = SingleEntityPhiTensor(
    child=child1, entity=ent, max_vals=upper1, min_vals=low1
)
# same entity, same data
tensor2 = SingleEntityPhiTensor(
    child=child1, entity=ent, max_vals=upper1, min_vals=low1
)
# same entity, different data
tensor3 = SingleEntityPhiTensor(
    child=child2, entity=ent, max_vals=upper2, min_vals=low2
)
# different entity, same data
tensor4 = SingleEntityPhiTensor(
    child=child1, entity=ent2, max_vals=upper1, min_vals=low1
)
# different entity, different data
tensor5 = SingleEntityPhiTensor(
    child=child2, entity=ent2, max_vals=upper2, min_vals=low2
)


# helper function to replace the value at a given index of a single entity phi tensor
def change_elem(tensor, ind, val) -> SingleEntityPhiTensor:
    tensor.child[ind] = val
    return tensor


simple_type1 = randint(-6, -4)
simple_type2 = randint(4, 6)


def test_le() -> None:

    assert tensor1.__le__(tensor2).child.all()
    assert not tensor3.__le__(tensor1).child.all()
    assert tensor1.__le__(tensor4) == NotImplemented
    assert tensor1.__le__(tensor5) == NotImplemented
    assert not tensor1.__le__(simple_type1).child.all()
    assert tensor1.__le__(simple_type2).child.all()


def test_ge() -> None:

    assert tensor1.__ge__(tensor2).child.all()
    assert not tensor1.__ge__(tensor3).child.all()
    assert tensor1.__ge__(tensor4) == NotImplemented
    assert tensor1.__ge__(tensor5) == NotImplemented
    assert tensor1.__ge__(simple_type1).child.all()
    assert not tensor1.__ge__(simple_type2).child.all()


def test_lt() -> None:

    assert not tensor1.__lt__(tensor2).child.all()
    assert tensor1.__lt__(tensor3).child.all()
    assert tensor1.__lt__(tensor4) == NotImplemented
    assert tensor1.__lt__(tensor5) == NotImplemented
    assert not tensor1.__lt__(simple_type1).child.all()
    assert tensor1.__lt__(simple_type2)


def test_gt() -> None:

    assert not tensor1.__gt__(tensor2).child.all()
    assert not tensor1.__gt__(tensor3).child.all()
    assert tensor1.__gt__(tensor4) == NotImplemented
    assert tensor1.__gt__(tensor5) == NotImplemented
    assert tensor1.__gt__(simple_type1).child.all()
    assert not tensor1.__gt__(simple_type2).child.all()


rand1 = np.random.randint(-4, 1)
rand2 = np.random.randint(1, 5)
clipped_tensor1 = tensor1.clip(rand1, rand2).child
clipped_tensor2 = tensor1.clip(rand2, rand1).child
clipped_tensor3 = tensor1.clip(rand1, None).child


def test_clip() -> None:
    assert ((clipped_tensor1 >= rand1) & (clipped_tensor1 <= rand2)).all()
    assert (clipped_tensor2 == rand1).all()
    assert (clipped_tensor3 >= rand1).all()


tensor1_copy = tensor1.copy()
tensor3_copy = tensor3.copy()


def test_copy() -> None:
    assert (
        (tensor1_copy.child == tensor1.child).all()
        & (tensor1_copy.min_vals == tensor1.min_vals).all()
        & (tensor1_copy.max_vals == tensor1.max_vals).all()
    )
    assert not (tensor3_copy.child == change_elem(tensor3, 0, rand1).child).all()


indices = sample(range(dims), dims)
tensor1_take = tensor1.take(indices)


def test_take() -> None:
    for i in range(dims):
        assert tensor1_take.child[i] == tensor1.child[indices[i]]


tensor6 = SingleEntityPhiTensor(
    child=np.arange(dims * dims).reshape(dims, dims),
    entity=ent,
    max_vals=dims ** 2 - 1,
    min_vals=0,
)
tensor6_diagonal = tensor6.diagonal()


def test_diagonal() -> None:
    for i in range(dims):
        assert tensor6_diagonal.child[i] == tensor6.child[i][i]


#
# ######################### ADD ############################
#
# MADHAVA: this needs fixing
@pytest.mark.xfail
def test_add(x: Tensor) -> None:
    z = x + x
    assert isinstance(z, Tensor), "Add: Result is not a Tensor"
    assert (
        z.child.min_vals == 2 * x.child.min_vals
    ).all(), "(Add, Minval) Result is not correct"
    assert (
        z.child.max_vals == 2 * x.child.max_vals
    ).all(), "(Add, Maxval) Result is not correct"


# MADHAVA: this needs fixing
@pytest.mark.xfail
def test_single_entity_phi_tensor_serde(x: Tensor) -> None:

    blob = serialize(x.child)
    x2 = deserialize(blob)

    assert (x.child.min_vals == x2.min_vals).all()
    assert (x.child.max_vals == x2.max_vals).all()


# def test_add(x,y):
#     z = x+y
#     assert isinstance(z, Tensor), "Add: Result is not a Tensor"
#     assert z.child.min_vals == x.child.min_vals + y.child.min_vals, "(Add, Minval) Result is not correct"
#     assert z.child.max_vals == x.child.max_vals + y.child.max_vals, "(Add, Maxval) Result is not correct"
#
# ######################### SUB ############################
#
# def test_sub(x):
#     z=x-x
#     assert isinstance(z, Tensor), "Sub: Result is not a Tensor"
#     assert z.child.min_vals == 0 * x.child.min_vals, "(Sub, Minval) Result is not correct"
#     assert z.child.max_vals == 0 * x.child.max_vals, "(Sub, Maxval) Result is not correct"
#
# def test_sub(x,y):
#     z=x-y
#     assert isinstance(z, Tensor), "Sub: Result is not a Tensor"
#     assert z.child.min_vals == x.child.min_vals - y.child.min_vals, "(Sub, Minval) Result is not correct"
#     assert z.child.max_vals == x.child.max_vals - y.child.max_vals, "(Sub, Maxval) Result is not correct"
#
# ######################### MUL ############################
#
# def test_mul(x):
#     z = x*x
#     assert isinstance(z, Tensor), "Mul: Result is not a Tensor"
#     assert z.child.min_vals == x.child.min_vals ** 2, "(Mul, Minval) Result is not correct"
#     assert z.child.max_vals == x.child.max_vals ** 2, "(Mul, Maxval) Result is not correct"
#
# def test_mul(x,y):
#     z = x*y
#     assert isinstance(z, Tensor), "Mul: Result is not a Tensor"
#     assert z.child.min_vals == x.child.min_vals ** 2, "(Mul, Minval) Result is not correct"
#     assert z.child.max_vals == x.child.max_vals ** 2, "(Mul, Maxval) Result is not correct"
