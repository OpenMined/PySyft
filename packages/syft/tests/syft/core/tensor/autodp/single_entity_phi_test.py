import pytest
from syft.core.tensor.tensor import Tensor
from syft.core.adp.entity import Entity
import numpy as np

gonzalo = Entity(name="Gonzalo")

############## general simple test fixture ###############

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

############## special fixtures for mul ##################
# Without loss of generality: term 1 <= term 2 mostly

@pytest.fixture(scope="function")
def pos_pos_1():
    # positive minval and positive maxval
    x = Tensor(np.array([[7]]))
    x = x.private(min_val=2, max_val=8, entity=gonzalo)
    return x

@pytest.fixture(scope="function")
def pos_pos_2():
    # positive minval and positive maxval
    x = Tensor(np.array([[6]]))
    x = x.private(min_val=5, max_val=10, entity=gonzalo)
    return x

@pytest.fixture(scope="function")
def neg_neg_1():
    # negative min_val and negative max_val
    x = Tensor(np.array([-5]))
    x = x.private(min_val=-6, max_val=-2, entity=gonzalo)
    return x

@pytest.fixture(scope="function")
def neg_neg_2():
    # negative min_val and negative max_val
    x = Tensor(np.array([-1]))
    x = x.private(min_val=-3, max_val=0, entity=gonzalo)
    return x

# Kritika: figure out all marginal cases of pos-neg * pos-neg
# @pytest.fixture(scope="function")
# def neg_pos_1():
#     x = Tensor(np.array(-1))
#     x = x.private(min_val=-2, max_val=4, entity=gonzalo)
#     return x

# @pytest.fixture(scope="function")
# def neg_pos_2():
#     x = Tensor(np.array(-1))
#     x = x.private(min_val=-1, max_val=6, entity=gonzalo)
#     return x

# @pytest.fixture(scope="function")
# def neg_pos_4():
#     x = Tensor(np.array(-1))
#     x = x.private(min_val=-6, max_val=1, entity=gonzalo)
#     return x

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

def test_mul_diff(x,y):
    z = x*y
    assert isinstance(z, Tensor), "Mul: Result is not a Tensor"
    min_min = x.child.min_vals * y.child.min_vals
    min_max = x.child.min_vals * y.child.max_vals
    max_min = x.child.max_vals * y.child.min_vals
    max_max = x.child.max_vals * y.child.max_vals
    # using .reduce as np.minimum takes in 2 ndarrays as input
    min_val = np.minimum.reduce([min_min, min_max, max_min, max_max])
    max_val = np.maximum.reduce([min_min, min_max, max_min, max_max])
    assert (z.child.min_vals == min_val).all(), "(Mul, Minval) Result is not correct"
    assert (z.child.max_vals == max_val).all(), "(Mul, Maxval) Result is not correct"

#     test if metadata is right
#     sign combinations

def test_mul_diff_pos_pos(pos_pos_1, pos_pos_2):
    z = pos_pos_1 * pos_pos_2
    assert isinstance(z, Tensor), "Mul: Result is not a Tensor"
    min_val = pos_pos_1.child.min_vals * pos_pos_2.child.min_vals
    max_val = pos_pos_1.child.max_vals * pos_pos_2.child.max_vals
    assert (z.child.min_vals == min_val).all(), "(Mul, Minval) Result is not correct"
    assert (z.child.max_vals == max_val).all(), "(Mul, Maxval) Result is not correct"

def test_mul_diff_neg_neg(neg_neg_1, neg_neg_2):
    z = neg_neg_1 * neg_neg_2
    assert isinstance(z, Tensor), "Mul: Result is not a Tensor"
    min_val = neg_neg_1.child.max_vals * neg_neg_2.child.max_vals
    max_val = neg_neg_1.child.min_vals * neg_neg_2.child.min_vals
    assert (z.child.min_vals == min_val).all(), "(Mul, Minval) Result is not correct"
    assert (z.child.max_vals == max_val).all(), "(Mul, Maxval) Result is not correct"