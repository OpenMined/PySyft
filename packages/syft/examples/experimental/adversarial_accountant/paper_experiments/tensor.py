import numpy as np
import torch as th
from passthrough import PassthroughTensor
from passthrough import implements
from passthrough import inputs2child
from ancestors import AutogradTensorAncestor
from ancestors import SingleEntityPhiTensorAncestor

class Tensor(PassthroughTensor, AutogradTensorAncestor, SingleEntityPhiTensorAncestor):

    def __init__(self, child):
        """data must be a list of numpy array"""
        
        if isinstance(child, list):
            child = np.array(child)
        
        if isinstance(child, th.Tensor):
            child = child.numpy()
            
        if not isinstance(child, PassthroughTensor) and not isinstance(child, np.ndarray):
            raise Exception("Data must be list or nd.array")
        
        super().__init__(child=child)

        
@implements(Tensor, np.mean)
def mean(*args, **kwargs):
    print("meaning")
    args, kwargs = inputs2child(*args, **kwargs)
    return np.mean(*args, **kwargs)

@implements(Tensor, np.max)
def npmax(*args, **kwargs):
    print("maxing1")
    args, kwargs = inputs2child(*args, **kwargs)
    return np.max(*args, **kwargs)

@implements(Tensor, np.min)
def npmin(*args, **kwargs):
    print("mining1")
    args, kwargs = inputs2child(*args, **kwargs)
    return np.min(*args, **kwargs)

@implements(Tensor, np.square)
def square(x):
    return x*x