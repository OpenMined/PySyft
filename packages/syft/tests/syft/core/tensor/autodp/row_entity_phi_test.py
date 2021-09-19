# stdlib
from random import randint, uniform
from typing import List

# third party
import numpy as np
import pytest

# syft absolute
from syft import deserialize
from syft import serialize
from syft.core.adp.entity import Entity
from syft.core.tensor.autodp.row_entity_phi import RowEntityPhiTensor
from syft.core.tensor.autodp.single_entity_phi import SingleEntityPhiTensor
from syft.core.tensor.tensor import Tensor

ent = Entity(name="test")
ent2 = Entity(name="test2")

dims = np.random.randint(10) + 1
row_count = np.random.randint(10) + 1

def rept(low, high, entity) -> List:
    data = []
    for _ in range(row_count):
        data.append(
            SingleEntityPhiTensor(
                child = np.random.randint(low=low, high=high, size=dims),
                entity=entity,
                max_vals=np.full(dims, high-1, dtype=np.int32),
                min_vals=np.full(dims, low, dtype=np.int32)
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
    
    for i in tensor1.__le__(tensor2).child: assert i.child.all()
    for i in tensor1.__le__(tensor3).child: assert i.child.all()
    for i in tensor1.__le__(simple_type1).child: assert not i.child.all()
    for i in tensor1.__le__(simple_type2).child: assert i.child.all()


def test_ge() -> None:

    for i in tensor1.__ge__(tensor2).child: assert i.child.all()
    for i in tensor1.__ge__(tensor3).child: assert not i.child.all()
    for i in tensor1.__ge__(simple_type1).child: assert i.child.all()
    for i in tensor1.__ge__(simple_type2).child: assert not i.child.all()


def test_lt() -> None:

    for i in tensor1.__lt__(tensor2).child: assert not i.child.all()
    for i in tensor1.__lt__(tensor3).child: assert i.child.all()
    for i in tensor1.__lt__(simple_type1).child: assert not i.child.all()
    for i in tensor1.__lt__(simple_type2).child: assert i.child.all()



def test_gt() -> None:

    for i in tensor1.__gt__(tensor2).child: assert not i.child.all()
    for i in tensor1.__gt__(tensor3).child: assert not i.child.all()
    for i in tensor1.__gt__(simple_type1).child: assert i.child.all()
    for i in tensor1.__gt__(simple_type2).child: assert not i.child.all()