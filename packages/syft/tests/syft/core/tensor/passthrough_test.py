# third party

import numpy as np
import torch

# syft absolute
from syft.core.tensor.passthrough import PassthroughTensor



def test_data_child() -> None:
    data = np.array([1,2,3], dtype=np.int32)
    tensor = PassthroughTensor(child=data)

    assert (tensor._data_child == data).all()


def test_len() -> None:
    data_list = [1.5,3,True,"Thanos"]
    data_array = np.array([[1,2],[3,4]], dtype=np.int32)
    for i in data_list:
        if i == float or int or bool:
            data = np.array([i])
            tensor = PassthroughTensor(child=data)

            assert tensor.__len__() == 1
        
        else:
            data = np.array([i])
            tensor = PassthroughTensor(child=data)

            assert len(tensor) == 1

    tensor = PassthroughTensor(child=data_array)
            
    assert len(tensor) == 2


def test_shape() -> None:
    data_list = [1.5,3,True,"Thanos"]
    data_array = np.array([[1,2],[3,4]], dtype=np.int32)
    for i in data_list:
        if i == float or int or bool:
            data = np.array([i])
            tensor = PassthroughTensor(child=data)
            
            assert tensor.shape == (1,)
    
        else:
            data = np.array([i])
            tensor = PassthroughTensor(child=data)

            assert tensor.shape == (1,)

    tensor = PassthroughTensor(child=data_array)
            
    assert tensor.shape == (2,2)


def test_dtype() -> None:
    data = np.array([1,2,3], dtype=np.int32)
    tensor = PassthroughTensor(child=data)

    assert tensor.dtype == np.int32




# NEED HELP HERE Mr is_acceptable_simple_type
#
# I think understanding how to compare the two tensor arrays will solve a few of the
# issues I'm having.  Is this method meant to even compare two arrays?
# Am I just suppose to throw in a single value to compare against an array to 
# get a boolean array returned. 
# Also probably have to figure how to compare two arrays element wise, all and any, 
# Ive tried reading doucmentation in numpy and torch but it seems the way to compare
# arrays here uses a slightly different syntax than when dealing with numpy and torch arrays
# and I can't figure out the same fuctionallity with the custom tensor objects
# we have here.



# def test_logical_and() -> None:
#     input = np.array([False, True])
#     other = np.array([True, True])

#     tensor1 = PassthroughTensor(child=input)
#     tensor2 = PassthroughTensor(child=other)

#     x = (tensor1.logical_and(tensor2)).all()
  

#     print(x)

#     assert False




def test__abs__() -> None:
    data = np.array([1,-1,-2], dtype=np.int32)
    check = np.array([1,1,2], dtype=np.int32)
    tensor_a = PassthroughTensor(child=data)
    tensor_b = PassthroughTensor(child=check)
    
    assert tensor_a.__abs__() == tensor_b


def test__add__() -> None:
    data_a = np.array([1,-1,-2], dtype=np.int32)
    data_b = np.array([1,1,3], dtype=np.int32)
    data_c = np.array([2,0,1], dtype=np.int32)
    tensor_a = PassthroughTensor(child=data_a)
    tensor_b = PassthroughTensor(child=data_b)
    tensor_c = PassthroughTensor(child=data_c)
    result_a = tensor_a.__add__(tensor_b)
    result_b = tensor_a.__add__(data_b) 
    
    assert result_a == tensor_c and result_b == tensor_c
    

def test__radd__() -> None:
    data_a = np.array([1,-1,-2], dtype=np.int32)
    data_b = torch.tensor([1,1,3], dtype=torch.int32)
    data_c = torch.tensor([2,0,1], dtype=torch.int32)
    tensor_a = PassthroughTensor(child=data_a)
    tensor_b = PassthroughTensor(child=data_b)
    tensor_c = PassthroughTensor(child=data_c)
    result_a = tensor_a.__radd__(tensor_b)
    result_b = tensor_a.__radd__(data_b)
    
    assert  result_a == tensor_c and result_b == tensor_c 


def test__sub__() -> None:
    data_a = np.array([1,-1,-2], dtype=np.int32)
    data_b = np.array([1,1,3], dtype=np.int32)
    data_c = np.array([0,-2,-5], dtype=np.int32)
    tensor_a = PassthroughTensor(child=data_a)
    tensor_b = PassthroughTensor(child=data_b)
    tensor_c = PassthroughTensor(child=data_c)
    result_a = tensor_a.__sub__(tensor_b)
    result_b = tensor_a.__sub__(data_b)

    assert  result_a == tensor_c and result_b == tensor_c


def test__rsub__() -> None:
    data_a = np.array([1,-1,-2], dtype=np.int32)
    data_b = torch.tensor([1,1,3], dtype=torch.int32)
    data_c = torch.tensor([0,-2,-5], dtype=torch.int32)
    tensor_a = PassthroughTensor(child=data_a)
    tensor_b = PassthroughTensor(child=data_b)
    tensor_c = PassthroughTensor(child=data_c)
    result_a = tensor_b.__rsub__(tensor_a)
    result_b = tensor_b.__rsub__(data_a)

    assert  result_a == tensor_c and result_b == tensor_c 


def test__gt__() -> None:
    data_a = np.array([1,2,3], dtype=np.int32)
    data_b = np.array([0,3,3], dtype=np.int32)
    data_c = np.array([True, False, False])
    tensor_a = PassthroughTensor(child=data_a)
    tensor_b = PassthroughTensor(child=data_b)
    tensor_c = PassthroughTensor(child=data_c)
    result_a = tensor_a.__gt__(tensor_b)
    result_b = tensor_a.__gt__(data_b)

    assert result_a == tensor_c and result_b == tensor_c


def test__ge__() -> None:
    data_a = np.array([1,2,3], dtype=np.int32)
    data_b = np.array([0,3,3], dtype=np.int32)
    data_c = np.array([True, False, True])
    tensor_a = PassthroughTensor(child=data_a)
    tensor_b = PassthroughTensor(child=data_b)
    tensor_c = PassthroughTensor(child=data_c)
    result_a = tensor_a.__ge__(tensor_b)
    result_b = tensor_a.__ge__(data_b)

    assert result_a == tensor_c and result_b == tensor_c


def test__lt__() -> None:
    data_a = np.array([1,2,3], dtype=np.int32)
    data_b = np.array([2,1,3], dtype=np.int32)
    data_c = np.array([True, False, False])
    tensor_a = PassthroughTensor(child=data_a)
    tensor_b = PassthroughTensor(child=data_b)
    tensor_c = PassthroughTensor(child=data_c)
    result_a = tensor_a.__lt__(tensor_b)
    result_b = tensor_a.__lt__(data_b)
    
    assert result_a == tensor_c and result_b == tensor_c


def test__le__() -> None:
    data_a = np.array([1,2,3], dtype=np.int32)
    data_b = np.array([0,3,3], dtype=np.int32)
    data_c = np.array([False, True, True])
    tensor_a = PassthroughTensor(child=data_a)
    tensor_b = PassthroughTensor(child=data_b)
    tensor_c = PassthroughTensor(child=data_c)
    result_a = tensor_a.__le__(tensor_b)
    result_b = tensor_a.__le__(data_b)

    assert result_a == tensor_c and result_b == tensor_c


def test__ne__() -> None:
    data_a = np.array([1, 2, 3], dtype=np.int32)
    tensor_a = PassthroughTensor(child=data_a)
    result_a = tensor_a.__ne__(tensor_a) 
    result_b = tensor_a.__ne__(data_a) 

    assert result_a == False and result_b == True


def test__eq__() -> None:
    data_a = np.array([1, 2, 3], dtype=np.int32)
    tensor_a = PassthroughTensor(child=data_a)
    result_a = tensor_a.__eq__(tensor_a)
    result_b = tensor_a.__eq__(data_a)

    assert result_a == True and result_b == False

def test__floordiv__() -> None:
    data_a = np.array([1,2,3], dtype=np.int32)
    data_b = np.array([1,5,4], dtype=np.int32)
    data_c = np.array([1,2,0], dtype=np.int32)
    tensor_a = PassthroughTensor(child=data_a)
    tensor_b = PassthroughTensor(child=data_b)
    tensor_c = PassthroughTensor(child=data_c)
    result_a = tensor_a.__floordiv__(tensor_b)
    result_b = tensor_a.__floordiv__(data_b)

    assert result_a == tensor_c and result_b == tensor_c


def test__rfloordiv__() -> None:
    data_a = np.array([1,2,3], dtype=np.int32)
    data_b = torch.tensor([1,5,4], dtype=torch.int32)
    data_c = torch.tensor([1,2,0], dtype=torch.int32)
    tensor_a = PassthroughTensor(child=data_a)
    tensor_b = PassthroughTensor(child=data_b)
    tensor_c = PassthroughTensor(child=data_c)
    result_a = tensor_a.__rfloordiv__(tensor_b)
    result_b = tensor_a.__rfloordiv__(data_b)

    assert result_a == tensor_c and result_b == tensor_c


def test__lshift__() -> None:
    data_a = np.array([1,2,-1], dtype=np.int32)
    data_b = np.array([0,2,1], dtype=np.int32)
    data_c = np.array([0,8,-2], dtype=np.int32)
    tensor_a = PassthroughTensor(child=data_a)
    tensor_b = PassthroughTensor(child=data_b)
    tensor_c = PassthroughTensor(child=data_c)
    result_a = tensor_a.__lshift__(tensor_b)
    result_b = tensor_a.__lshift__(data_b)

    assert result_a == tensor_c and result_b == tensor_c


def test__rlshift__() -> None:
    data_a = np.array([1,2,-1], dtype=np.int32)
    data_b = torch.tensor([0,2,1], dtype=torch.int32)
    data_c = torch.tensor([0,8,-2], dtype=torch.int32)
    tensor_a = PassthroughTensor(child=data_a)
    tensor_b = PassthroughTensor(child=data_b)
    tensor_c = PassthroughTensor(child=data_c)
    result_a = tensor_a.__rlshift__(tensor_b)
    result_b = tensor_a.__rlshift__(data_b)

    assert result_a == tensor_c and result_b == tensor_c


def test__rshift__() -> None:
    data_a = np.array([1,2,-1], dtype=np.int32)
    data_b = np.array([2,1,1], dtype=np.int32)
    data_c = np.array([0,1,-1], dtype=np.int32)
    tensor_a = PassthroughTensor(child=data_a)
    tensor_b = PassthroughTensor(child=data_b)
    tensor_c = PassthroughTensor(child=data_c)
    result_a = tensor_a.__rshift__(tensor_b)
    result_b = tensor_a.__rshift__(data_b)

    print(result_a)

    assert result_a == tensor_c and result_b == tensor_c


def test__rrshift__() -> None:
    data_a = np.array([1,2,-1], dtype=np.int32)
    data_b = torch.tensor([0,2,1], dtype=torch.int32)
    data_c = torch.tensor([0,8,-1], dtype=torch.int32)
    tensor_a = PassthroughTensor(child=data_a)
    tensor_b = PassthroughTensor(child=data_b)
    tensor_c = PassthroughTensor(child=data_c)
    result_a = tensor_a.__rrshift__(tensor_b)
    result_b = tensor_a.__rrshift__(data_b)

    assert result_a == tensor_c and result_b == tensor_c


def test__pow__() -> None:
    data_a = np.array([1,2,-1], dtype=np.int32)
    data_b = np.array([0,2,1], dtype=np.int32)
    data_c = np.array([1,4,-1], dtype=np.int32)
    tensor_a = PassthroughTensor(child=data_a)
    tensor_b = PassthroughTensor(child=data_b)
    tensor_c = PassthroughTensor(child=data_c)
    result_a = tensor_a.__pow__(tensor_b)
    result_b = tensor_a.__pow__(data_b)

    assert result_a == tensor_c and result_b == tensor_c

def test__rpow__() -> None:
    data_a = np.array([1,2,1], dtype=np.int32)
    data_b = torch.tensor([0,2,-1], dtype=torch.int32)
    data_c = torch.tensor([0,4,-1], dtype=torch.int32)
    tensor_a = PassthroughTensor(child=data_a)
    tensor_b = PassthroughTensor(child=data_b)
    tensor_c = PassthroughTensor(child=data_c)
    result_a = tensor_a.__rpow__(tensor_b)
    result_b = tensor_a.__rpow__(data_b)
    
    assert result_a == tensor_c and result_b == tensor_c


def test__divmod__() -> None:
    data_a = np.array([1,2,3], dtype=np.int32)
    data_b = np.array([1,5,4], dtype=np.int32)
    data_c = np.array([0,1,3], dtype=np.int32)
    tensor_a = PassthroughTensor(child=data_a)
    tensor_b = PassthroughTensor(child=data_b)
    tensor_c = PassthroughTensor(child=data_c)
    result_a = tensor_a.__divmod__(tensor_b)
    result_b = tensor_a.__divmod__(data_b)

    assert result_a == tensor_c and result_b == tensor_c


def test__neg__() -> None:
    data_a = np.array([1,0,-1], dtype=np.int32)
    data_b = np.array([-1,0,1], dtype=np.int32)
    tensor_a = PassthroughTensor(child=data_a)
    tensor_b = PassthroughTensor(child=data_b)

    assert tensor_a.__neg__() == tensor_b


def test__invert__() -> None:
    data_a = np.array([-1,0,1], dtype=np.int32)
    data_b = np.array([0,-1,-2], dtype=np.int32)
    tensor_a = PassthroughTensor(child=data_a)
    tensor_b = PassthroughTensor(child=data_b)
    tensor_c = PassthroughTensor(child=tensor_a.__invert__())

    assert tensor_b == tensor_c



# I have no clue if I implemented this right
def test__index__() -> None:
    data_a = np.array([1,2,3], dtype=np.int32)
    tensor_a = PassthroughTensor(child=data_a)

    assert tensor_a[2].__index__() == 3



def test_copy() -> None:
    data_a = np.array([1,2,3], dtype=np.int32)
    tensor_a = PassthroughTensor(child=data_a)

    assert tensor_a.copy() == tensor_a


def test__mul__() -> None:
    data_a = np.array([1,2,-1], dtype=np.int32)
    data_b = np.array([0,2,1], dtype=np.int32)
    data_c = np.array([0,4,-1], dtype=np.int32)
    tensor_a = PassthroughTensor(child=data_a)
    tensor_b = PassthroughTensor(child=data_b)
    tensor_c = PassthroughTensor(child=data_c)
    result_a = tensor_a.__mul__(tensor_b)
    result_b = tensor_a.__mul__(data_b)

    assert result_a == tensor_c and result_b == tensor_c


def test__rmul__() -> None:
    data_a = np.array([1,2,-1], dtype=np.int32)
    data_b = torch.tensor([0,2,1], dtype=torch.int32)
    data_c = torch.tensor([0,4,-1], dtype=torch.int32)
    tensor_a = PassthroughTensor(child=data_a)
    tensor_b = PassthroughTensor(child=data_b)
    tensor_c = PassthroughTensor(child=data_c)
    result_a = tensor_a.__rmul__(tensor_b)
    result_b = tensor_a.__rmul__(data_b)

    assert result_a == tensor_c and result_b == tensor_c


def test__matmul__() -> None:
    data_a = np.array([[1,2],[3,4]], dtype=np.int32)
    data_b = np.array([[1,2],[3,4]], dtype=np.int32)
    data_c = np.array([[7,10],[15,22]],  dtype=np.int32)
    tensor_a = PassthroughTensor(child=data_a)
    tensor_b = PassthroughTensor(child=data_c)

    assert tensor_a.__matmul__(data_b) == tensor_b


def test__rmatmul__() -> None:
    data_a = np.array([[1,2],[3,4]], dtype=np.int32)
    data_b = torch.tensor([[1,2],[3,4]], dtype=torch.int32)
    data_b = data_b.numpy()
    data_c = np.array([[7,10],[15,22]],  dtype=np.int32)
    tensor_a = PassthroughTensor(child=data_a)
    tensor_b = PassthroughTensor(child=data_b)
    tensor_c = PassthroughTensor(child=data_c)

    assert tensor_b.__rmatmul__(tensor_a) == tensor_c


def test__truediv__() -> None:
    data_a = np.array([1,2,-3], dtype=np.int32)
    data_b = np.array([1,3,2], dtype=np.int32)
    data_c = np.array([1.,0.66666667,-1.5], dtype=np.float32)
    tensor_a = PassthroughTensor(child=data_a)
    tensor_b = PassthroughTensor(child=data_b)
    tensor_c = PassthroughTensor(child=data_c)
    result_a = tensor_a.__truediv__(tensor_b)
    result_b = tensor_a.__truediv__(data_b)

    assert result_a == tensor_c and result_b == tensor_c


def test__rtruediv__() -> None:
    data_a = np.array([1,2,3], dtype=np.int32)
    data_b = torch.tensor([1.,3.,-3.], dtype=torch.float32)
    data_c = torch.tensor([1.,1.5,-1.], dtype=torch.float32)
    tensor_a = PassthroughTensor(child=data_a)
    tensor_b = PassthroughTensor(child=data_b)
    tensor_c = PassthroughTensor(child=data_c)
    result = tensor_a.__rtruediv__(tensor_b)

    assert result == tensor_c


def test_manual_dot() -> None:
    data_a = np.array([[1,2],[3,4]], dtype=np.int32)
    data_b = np.array([[1,2],[3,4]], dtype=np.int32)
    data_c = np.array([[7,10],[15,22]],  dtype=np.int32)
    tensor_a = PassthroughTensor(child=data_a)
    tensor_b = PassthroughTensor(child=data_c)

    assert tensor_a.manual_dot(data_b) == tensor_b


def test_dot() -> None:
    data_a = np.array([[1,2],[3,4]], dtype=np.int32)
    data_b = np.array([[1,2],[3,4]], dtype=np.int32)
    data_c = np.array([[7,10],[15,22]],  dtype=np.int32)
    tensor_a = PassthroughTensor(child=data_a)
    tensor_b = PassthroughTensor(child=data_c)

    assert tensor_a.dot(data_b) == tensor_b


def test_reshape() -> None:
    data_a = np.array([[1,2],[3,4]], dtype=np.int32)
    data_b = np.array([1,2,3,4], dtype=np.int32)
    tensor_a = PassthroughTensor(data_a)
    tensor_b = PassthroughTensor(data_b)
    result = tensor_a.reshape((1,4))

    assert result == tensor_b


def test_repeat() -> None:
    data_a = np.array([[1,2],[3,4]], dtype=np.int32)
    tensor_a = PassthroughTensor(data_a)
    print(tensor_a.repeat(2, axis=1))

    assert False

