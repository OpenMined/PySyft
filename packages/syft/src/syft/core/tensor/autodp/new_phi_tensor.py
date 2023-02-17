import numpy as np
from ..lazy_repeat_array import lazyrepeatarray
from ...adp.data_subject import DataSubject
from .phi_tensor import PhiTensor
from .gamma_tensor import GammaTensor
from typing import List
from typing import Optional
from typing import Union
from typing import Tuple
from typing import Any
from typing import Dict
from ...common.uid import UID
from ..passthrough import is_acceptable_simple_type
from ..broadcastable import is_broadcastable
from .jax_ops import SyftTerminalNoop

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

class BoundsArray():
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    shape: Tuple[int, ...]

    def __init__(self, lower_bounds, upper_bounds, shape):
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.shape = shape
    
    @staticmethod
    def from_phi_tensors(phi_tensors):
        shape = (len(phi_tensors), *phi_tensors[0].shape)
        lower_bounds = np.array([phi_tensor.min_vals.data for phi_tensor in phi_tensors])
        upper_bounds = np.array([phi_tensor.max_vals.data for phi_tensor in phi_tensors])
        return BoundsArray(lower_bounds=lower_bounds, upper_bounds=upper_bounds, shape=shape)

    def __add__(self, other: Any):
        if is_acceptable_simple_type(other):
            return self.__class__(
                    lower_bounds=self.lower_bounds + other, 
                    upper_bounds=self.upper_bounds + other, 
                    shape=self.shape
                )

        if not is_broadcastable(self.shape, other.shape):
            raise Exception(
                f"Cannot broadcast arrays with shapes: {self.shape} & {other.shape}"
            )

        # if self.data.shape == other.data.shape:
        return self.__class__(
                lower_bounds=self.lower_bounds + other.lower_bounds, 
                upper_bounds=self.upper_bounds + other.upper_bounds, 
                shape=self.shape
            )
        # else:
        #     return self.__class__(data=self.data + other.data, shape=self.shape)

    def sum(self, axis: Optional[Union[int, Tuple[int, ...]]]):
        shape = ()
        prod_shape = np.prod(self.shape)
        lower_bounds = np.sum(self.lower_bounds * prod_shape)
        upper_bounds = np.sum(self.upper_bounds * prod_shape) 
        return BoundsArray(lower_bounds=lower_bounds,upper_bounds=upper_bounds,shape=shape)

    def to_numpy(self) -> np.ndarray:
        # FIX: shape is not set sometimes
        if not self.shape:
            self.shape = self.data.shape

        return np.broadcast_to(self.data, self.shape)

    def delete(self, indexes, axis):
        raise NotImplemented

    @staticmethod
    def compress_data(min_data, max_data):
        
        return BoundsArray(...)

def is_op_linear(op):
    if op == "sum":
        return True
    return False


class RowPhiTensors():
    data: np.array
    data_subjects: List[DataSubject]
    bounds: BoundsArray
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
        bounds = BoundsArray.from_phi_tensors(phi_tensors)
        # self.source_phi_tensors = phi_tensors
        return RowPhiTensors(data=data, data_subjects=data_subjects, bounds=bounds)
    
    def broadcast_op(self, op, *args, **kwargs):
        new_data = getattr(self.data, op)(*args, **kwargs)
        new_bounds = getattr(self.bounds, op)(*args, **kwargs)
        return RowPhiTensors(data=new_data, bounds=new_bounds, data_subjects=self.data_subjects)
   
    def reduce_op(self, op, *args, **kwargs):
        new_data = getattr(self.data, op)(*args, **kwargs)
        is_linear = is_op_linear(op)
        return GammaTensor(child=new_data, jax_op=None, sources={self.id}, is_linear=is_linear)
    
    def create_gamma(self):
        jax_op = SyftTerminalNoop(phi_id=self.id)
        return GammaTensor(
            child=self.data,
            jax_op=jax_op,
            sources={self.id},
            is_linear=True
        )

    def reconstruct(self, state: Dict[str, np.ndarray]) -> np.ndarray:
        return state[self.id] 

    def filtered(self, data_subjects):
        # TODO: Optimize this???
        data_subjects_indexes = [
            self.data_subjects.index(data_subject) for data_subject in data_subjects
        ]
        new_data_subjects = []
        for data_subject in self.data_subjects:
            if data_subject not in data_subjects:
                new_data_subjects.append(data_subject)
        
        new_data = np.delete(self.data, data_subjects_indexes, 0)
        new_bounds = self.bounds.delete(data_subjects_indexes, 0)
        
        return RowPhiTensors(
            data=new_data,
            data_subjects=new_data_subjects,
            bounds=new_bounds
        )
    
    def __add__(self, other):
        return self.broadcast_op('__add__', other)
    
    def sum(self, axis: Optional[Union[int, Tuple[int, ...]]]):
        return self.reduce_op("sum", axis)
    
class FlexibleMultiPhiTensors():
    
    def __init__(self) -> None:
        pass
    