# third party
import numpy as np

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


def test_logical_and() -> None:
    input = np.array([0,1,2,3], dtype=np.int32)
    other = np.array([0,7,0,9], dtype=np.int32)

    tensor1 = PassthroughTensor(child=input)
    tensor2 = PassthroughTensor(child=other)

    x = (tensor1.logical_and(tensor2)).any()

    print(x)

    assert False