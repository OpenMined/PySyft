import numpy as np
import syft.controller
from .float_tensor import FloatTensor

class MPCTensor():
    def __init__(self, data, data_as_tensor = False):
        if data_as_tensor:
            self.data = data
        else:
            self.data = FloatTensor(data)
    def shard(self):
        rand_tensor = self.data.random_()
        self.data = self.data-rand_tensor
        return rand_tensor
    def __add__(self,x):
        return self.data+x.data
