from abc import ABC, abstractmethod
import torch

class AbstractTensor(ABC):
    """
    This is the tensor abstraction
    """

    def wrap(self):
        wrapper = torch.Tensor()
        wrapper.child = self
        return wrapper