import syft
import torch

from typing import Union
from syft.decorators import syft_decorator

class DataTensor():
    
    @syft_decorator(typechecking=True)
    def __init__(self, child: Union[torch.FloatTensor, torch.IntTensor]):
        self.child=child
    
    def __add__(self, other):
        return DataTensor(child=self.child + other.child)

class FloatTensor():
    
    @syft_decorator(typechecking=True)
    def __init__(self, child: DataTensor):
        self.child=child
        
    def __add__(self, other):
        return FloatTensor(child=self.child + other.child)
    
class IntegerTensor():
    
    @syft_decorator(typechecking=True)
    def __init__(self, child: DataTensor):
        self.child=child
        
    def __add__(self, other):
        return FloatTensor(child=self.child + other.child)

class SyftTensor():
    def __init__(self, child: Union[FloatTensor, IntegerTensor]):
        self.child = child

    def __add__(self, other):
        return SyftTensor(child=self.child + other.child)
    
    @classmethod
    def FloatTensor(cls, data):
        if isinstance(data, list):
            return cls(child=FloatTensor(child=DataTensor(child=torch.FloatTensor(data))))