import numpy as np
import syft as sy
from typing import List


class Bounds(np.lib.mixins.NDArrayOperatorsMixin):
    lower_bound: int
    upper_bound: int
    
    def __init__(self, lower_bound: int, upper_bound:int):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(lower_bound={self.lower_bound}, upper_bound={self.upper_bound})'

    def __array__(self, dtype=None):
        return np.array([self.lower_bound, self.upper_bound])


class PhiTensor(np.lib.mixins.NDArrayOperatorsMixin):
    data: np.array
    bounds: Bounds
    data_subject: int
    
    def __init__(self, data: np.ndarray, lower_bound: int, upper_bound: int, data_subject:str) -> None:
        self.data = data
        self.bounds = Bounds(lower_bound=lower_bound, upper_bound=upper_bound)
        self.data_subject = data_subject
        
    def __repr__(self) -> str:
        return f'{self.__class__.__name__},{self.bounds.__repr__()}, ds={self.data_subject}'
        
    def __array__(self):
        return self.data

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        new_data = 

class PhiTensorCollection:
    phi_tensors: List[PhiTensor]
    data: np.array
    
    def __init__(self, phi_tensors: List[PhiTensor]) -> None:
        self.phi_tensors = phi_tensors
        self.data = np.array(phi_tensors)
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(phi_tensors=({len(self.phi_tensors)}))'#{"/".join([pt.__repr__() for pt in self.phi_tensors])})'