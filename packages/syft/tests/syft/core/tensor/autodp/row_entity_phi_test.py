# stdlib
from random import randint
from random import sample

# third party
import numpy as np

# syft absolute
from syft.core.adp.entity import Entity
from syft.core.tensor.autodp.row_entity_phi import RowEntityPhiTensor
from syft.core.tensor.autodp.single_entity_phi import SingleEntityPhiTensor

ent = Entity(name="test")
ent2 = Entity(name="test2")

dims = np.random.randint(10) + 1
row_count = np.random.randint(10) + 1


def rept(low, high, entity) -> RowEntityPhiTensor:
    data = []
    for _ in range(row_count):
        data.append(
            SingleEntityPhiTensor(
                child=np.random.randint(low=low, high=high, size=dims),
                entity=entity,
                max_vals=np.full(dims, high - 1, dtype=np.int32),
                min_vals=np.full(dims, low, dtype=np.int32),
            )
        )
    return RowEntityPhiTensor(rows=data, check_shape=False)


tensor1 = rept(-2, 4, ent)
# same entity, same data
tensor2 = tensor1
# same entity, different data
tensor3 = rept(4, 7, ent)

simple_type1 = randint(-6, -4)
simple_type2 = randint(4, 6)


def test_le() -> None:

    for i in tensor1.__le__(tensor2).child:
        assert i.child.all()
    for i in tensor1.__le__(tensor3).child:
        assert i.child.all()
    for i in tensor1.__le__(simple_type1).child:
        assert not i.child.all()
    for i in tensor1.__le__(simple_type2).child:
        assert i.child.all()


def test_ge() -> None:

    for i in tensor1.__ge__(tensor2).child:
        assert i.child.all()
    for i in tensor1.__ge__(tensor3).child:
        assert not i.child.all()
    for i in tensor1.__ge__(simple_type1).child:
        assert i.child.all()
    for i in tensor1.__ge__(simple_type2).child:
        assert not i.child.all()


def test_lt() -> None:

    for i in tensor1.__lt__(tensor2).child:
        assert not i.child.all()
    for i in tensor1.__lt__(tensor3).child:
        assert i.child.all()
    for i in tensor1.__lt__(simple_type1).child:
        assert not i.child.all()
    for i in tensor1.__lt__(simple_type2).child:
        assert i.child.all()


def test_gt() -> None:

    for i in tensor1.__gt__(tensor2).child:
        assert not i.child.all()
    for i in tensor1.__gt__(tensor3).child:
        assert not i.child.all()
    for i in tensor1.__gt__(simple_type1).child:
        assert i.child.all()
    for i in tensor1.__gt__(simple_type2).child:
        assert not i.child.all()


rand1 = np.random.randint(-4, 1)
rand2 = np.random.randint(1, 5)
clipped_tensor1 = tensor1.clip(rand1, rand2).child
clipped_tensor2 = tensor1.clip(rand2, rand1).child
clipped_tensor3 = tensor1.clip(rand1, None).child


def test_clip() -> None:
    for i in clipped_tensor1:
        assert ((i.child >= rand1) & (i.child <= rand2)).all()
    for i in clipped_tensor2:
        assert (i.child == rand1).all()
    for i in clipped_tensor3:
        assert (i.child >= rand1).all()


tensor1_copy = tensor1.copy()


def test_copy() -> None:
    for i in range(len(tensor1)):
        assert (
            (tensor1_copy[i].child.child == tensor1[i].child.child).all()
            & (tensor1_copy[i].child.min_vals == tensor1[i].child.min_vals).all()
            & (tensor1_copy[i].child.max_vals == tensor1[i].child.max_vals).all()
        )


indices = sample(range(dims), dims)
tensor1_take = tensor1.take(indices)


def test_take() -> None:
    for i in range(len(tensor1)):
        for j in range(dims):
            assert tensor1_take.child[i].child[j] == tensor1.child[i].child[indices[j]]


def rept_with_sq_sept(num_rows) -> RowEntityPhiTensor:
    new_list = list()
    for _ in range(num_rows):
        new_list.append(
            SingleEntityPhiTensor(
                child=np.arange(dims * dims).reshape(dims, dims),
                entity=ent,
                max_vals=np.full((dims, dims), dims ** 2 - 1),
                min_vals=np.full((dims, dims), 0),
            )
        )

    return RowEntityPhiTensor(rows=new_list, check_shape=False)


num_rows = np.random.randint(10) + 1
tensor4 = rept_with_sq_sept(num_rows)
tensor4_diagonal = tensor4.diagonal()


def test_diagonal() -> None:
    for i in range(len(tensor4)):
        for j in range(dims):
            assert tensor4_diagonal.child[i].child[j] == tensor4.child[i].child[j][j]
