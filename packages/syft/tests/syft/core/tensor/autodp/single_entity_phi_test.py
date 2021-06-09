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

andrew = Entity(name="Andrew")
chinmay = Entity(name="Chinmay")
george = Entity(name="George")
kritika = Entity(name="Kritika")
madhava = Entity(name="Madhava")
tudor = Entity(name="Tudor")

@pytest.fixture(scope="function")
def weather():
    w = Tensor(np.array[2,14,-20,50,21,32,-6,20,17,24,10,24,-20,40])
    # min_vals = np.array([0,6,-62,-13,12,21,-10,6,12,21,5,12,-38,-7])
    # max_vals = np.array([11,20,38,57,28,36,20,35,28,36,22,28,26,45])
    entities = [andrew, andrew, andrew, andrew, chinmay, chinmay, 
                george, george, kritika, kritika, madhava, madhava, 
                tudor, tudor]
    w = w.private(min_val=-20, max_val=57, entities=entities)
    # x= x.private(min_val=min_vals, max_val=max_vals, entities=entities)
    return w

#
# ######################### ADD ############################
#
def test_add(x):
    z = x+x
    assert isinstance(z, Tensor), "Add: Result is not a Tensor"
    assert (z.child.min_vals == 2 * x.child.min_vals).all(), "(Add, Minval) Result is not correct"
    assert (z.child.max_vals == 2 * x.child.max_vals).all(), "(Add, Maxval) Result is not correct"

def test_add_diff(x,y):
    z = x+y
    assert isinstance(z, Tensor), "Add: Result is not a Tensor"
    assert z.child.min_vals == x.child.min_vals + y.child.min_vals, "(Add, Minval) Result is not correct"
    assert z.child.max_vals == x.child.max_vals + y.child.max_vals, "(Add, Maxval) Result is not correct"
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
