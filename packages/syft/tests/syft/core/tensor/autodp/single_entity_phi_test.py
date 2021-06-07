from syft.core.tensor.tensor import Tensor
from syft.core.adp.entity import Entity
import numpy as np

gonzalo = Entity(name="Gonzalo")

x = Tensor(np.array([[1,2,3],[4,5,6]]))
x = x.private(min_val=-1,max_val=7,entity=gonzalo)
y = Tensor(np.array([[-1,-2,-3],[-4,-5, -6]]))
y = y.private(min_val=-7,max_val=1,entity=gonzalo)

######################### ADD ############################

def test_add(x):
    z = x+x
    assert isinstance(z, Tensor), "Add: Result is not a Tensor"
    assert z.child.min_vals == 2 * x.child.min_vals, "(Add, Minval) Result is not correct"
    assert z.child.max_vals == 2 * x.child.max_vals, "(Add, Maxval) Result is not correct"
    
def test_add(x,y):
    z = x+y
    assert isinstance(z, Tensor), "Add: Result is not a Tensor"
    assert z.child.min_vals == x.child.min_vals + y.child.min_vals, "(Add, Minval) Result is not correct"
    assert z.child.max_vals == x.child.max_vals + y.child.max_vals, "(Add, Maxval) Result is not correct"
    
######################### SUB ############################
    
def test_sub(x):
    z=x-x
    assert isinstance(z, Tensor), "Sub: Result is not a Tensor"
    assert z.child.min_vals == 0 * x.child.min_vals, "(Sub, Minval) Result is not correct"
    assert z.child.max_vals == 0 * x.child.max_vals, "(Sub, Maxval) Result is not correct"
    
def test_sub(x,y):
    z=x-y
    assert isinstance(z, Tensor), "Sub: Result is not a Tensor"
    assert z.child.min_vals == x.child.min_vals - y.child.min_vals, "(Sub, Minval) Result is not correct"
    assert z.child.max_vals == x.child.max_vals - y.child.max_vals "(Sub, Maxval) Result is not correct"
    
######################### MUL ############################
    
def test_mul(x):
    z = x*x
    assert isinstance(z, Tensor), "Mul: Result is not a Tensor"
    assert z.child.min_vals == x.child.min_vals ** 2, "(Mul, Minval) Result is not correct"
    assert z.child.max_vals == x.child.max_vals ** 2, "(Mul, Maxval) Result is not correct"
    
def test_mul(x,y):
    z = x*y
    assert isinstance(z, Tensor), "Mul: Result is not a Tensor"
    assert z.child.min_vals == x.child.min_vals ** 2, "(Mul, Minval) Result is not correct"
    assert z.child.max_vals == x.child.max_vals ** 2, "(Mul, Maxval) Result is not correct"

    
######################### main ############################
    
if __name__ == "__main__":
    test_add(x)
    test_add(x,y)
    test_sub(x)
    test_sub(x,y)
    test_mul(x)
    test_mul(x,y)