import numpy as np
from ..lazy_repeat_array import lazyrepeatarray
from ...adp.data_subject import DataSubject
from .phi_tensor import PhiTensor
from .gamma_tensor import GammaTensor
from typing import List
from typing import Optional
from typing import Union
from typing import Tuple
from ...common.uid import UID

class Bounds():
    lower_bound: int
    upper_bound: int
    
    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def __add__(self, other):
        if isinstance(other, Bounds):
            new_lower_bound = self.lower_bound + other.lower_bound
            new_upper_bound = self.upper_bound + other.upper_bound
            return Bounds(new_lower_bound, new_upper_bound)

class RowPhiTensors():
    data: np.array
    data_subjects: List[DataSubject]
    bounds: lazyrepeatarray
    # source_ids: List[PhiTensor]
    
    def __init__(
        self, 
        data, 
        data_subjects, 
        bounds,
        id: Optional[UID] = None,
    ) -> None:
        self.data = data
        self.data_subjects = data_subjects
        self.bounds = bounds
        if id is None:
            id = UID()
        self.id = id
    
    @staticmethod
    def from_phi_tensors(phi_tensors) -> None:
        data = np.array([phi_tensor.child for phi_tensor in phi_tensors])
        data_subjects = [phi_tensor.data_subject for phi_tensor in phi_tensors]
        bounds = [Bounds(phi_tensor.min_vals.data, phi_tensor.max_vals.data) for phi_tensor in phi_tensors]
        # self.source_phi_tensors = phi_tensors
        return RowPhiTensors(data=data, data_subjects=data_subjects, bounds=bounds)
    
    def broadcast_op(self, op, *args, **kwargs):
        new_data = getattr(self.data, op)(*args, **kwargs)
        # new_bounds = getattr(self.bounds, op)(*args, **kwargs)
        return RowPhiTensors(data=new_data, bounds=[], data_subjects=self.data_subjects)
    
    def reduce_op(self, op, *args, **kwargs):
        new_data = getattr(self.data, op)(*args, **kwargs)
        return GammaTensor(child=new_data, jax_op=None, sources={self.id})
    
    def __add__(self, other):
        return self.broadcast_op('__add__', other)
    
    def sum(self, axis: Optional[Union[int, Tuple[int, ...]]]):
        return self.reduce_op("sum", axis)