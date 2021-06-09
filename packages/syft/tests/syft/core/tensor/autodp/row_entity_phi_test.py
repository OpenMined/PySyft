import pytest
from syft.core.tensor.tensor import Tensor
from syft.core.adp.entity import Entity
import numpy as np

andrew = Entity(name="Andrew")
chinmay = Entity(name="Chinmay")
george = Entity(name="George")
kritika = Entity(name="Kritika")
madhava = Entity(name="Madhava")
tudor = Entity(name="Tudor")

@pytest.fixture(scope="function")
def x():
    x = Tensor(np.array[2,14,-20,50,21,32,-6,20,17,24,10,24,-20,40])
    # min_vals = np.array([0,6,-62,-13,12,21,-10,6,12,21,5,12,-38,-7])
    # max_vals = np.array([11,20,38,57,28,36,20,35,28,36,22,28,26,45])
    entities = [andrew, andrew, andrew, andrew, chinmay, chinmay, 
                george, george, kritika, kritika, madhava, madhava, 
                tudor, tudor]
    x = x.private(min_val=-20, max_val=57, entities=entities)
    # x= x.private(min_val=min_vals, max_val=max_vals, entities=entities)
    return x

#
# ######################### ADD ############################
#
# def test_add(x):
#     z = x+x
#     assert isinstance(z, Tensor), "Add: Result is not a Tensor"
#     assert (z.child.min_vals == 2 * x.child.min_vals).all(), "(Add, Minval) Result is not correct"
#     assert (z.child.max_vals == 2 * x.child.max_vals).all(), "(Add, Maxval) Result is not correct"