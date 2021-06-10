import pytest
from syft.core.tensor.tensor import Tensor
from syft.core.adp.entity import Entity
import numpy as np

gonzalo = Entity(name="Gonzalo")

@pytest.fixture(scope="function")
def x():
    x = Tensor(np.array([[1,2,3],[4,5,6]]))
    x = x.private(min_val=-1,max_val=7,entity=gonzalo)
    return x
@pytest.fixture(scope="function")
def y():
    y = Tensor(np.array([[-1,-2,-3],[-4,-5, -6]]))
    y = y.private(min_val=-7,max_val=1,entity=gonzalo)
    return y

######################### ADD ############################

def test_add(x):
    z = x+x
    assert isinstance(z, Tensor), "Add: Result is not a Tensor"
    assert (z.child.min_vals == 2 * x.child.min_vals).all(), "(Add, Minval) Result is not correct"
    assert (z.child.max_vals == 2 * x.child.max_vals).all(), "(Add, Maxval) Result is not correct"

def test_add_diff(x,y):
    z = x+y
    assert isinstance(z, Tensor), "Add: Result is not a Tensor"
    assert (z.child.min_vals == (x.child.min_vals + y.child.min_vals)).all(), "(Add, Minval) Result is not correct"
    assert (z.child.max_vals == (x.child.max_vals + y.child.max_vals)).all(), "(Add, Maxval) Result is not correct"

######################## SUB ############################

def test_sub(x):
    z=x-x
    assert isinstance(z, Tensor), "Sub: Result is not a Tensor"
    assert (z.child.min_vals == 0 * x.child.min_vals).all(), "(Sub, Minval) Result is not correct"
    assert (z.child.max_vals == 0 * x.child.max_vals).all(), "(Sub, Maxval) Result is not correct"

def test_sub_diff(x,y):
    z=x-y
    assert isinstance(z, Tensor), "Sub: Result is not a Tensor"
    assert (z.child.min_vals == x.child.min_vals - y.child.min_vals).all(), "(Sub, Minval) Result is not correct"
    assert (z.child.max_vals == x.child.max_vals - y.child.max_vals).all(), "(Sub, Maxval) Result is not correct"

######################## MUL ############################

def test_mul(x):
    z = x*x
    assert isinstance(z, Tensor), "Mul: Result is not a Tensor"

    sq_min_val = np.minimum(x.child.min_vals ** 2, x.child.max_vals ** 2)
    sq_max_val = np.maximum(x.child.min_vals ** 2, x.child.max_vals ** 2)
    assert (z.child.min_vals == sq_min_val).all(), "(Mul, Minval) Result is not correct"
    assert (z.child.max_vals == sq_max_val).all(), "(Mul, Maxval) Result is not correct"

# def test_mul_diff(x,y):
#     z = x*y
#     assert isinstance(z, Tensor), "Mul: Result is not a Tensor"
#     assert z.child.min_vals == x.child.min_vals ** 2, "(Mul, Minval) Result is not correct"
#     assert z.child.max_vals == x.child.max_vals ** 2, "(Mul, Maxval) Result is not correct"