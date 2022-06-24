# stdlib
from typing import Optional

# third party
import numpy as np
from torch import Tensor
from torch import nn

# relative
from ...adp.data_subject_list import DataSubjectList as DSL
from ..autodp.phi_tensor import PhiTensor
from .utils import dp_log
from .utils import dp_maximum


class Loss(object):
    """An objective function (or loss function, or optimization score 
    function) is one of the two parameters required to compile a model.
    
    """
    def forward(self, outputs: PhiTensor, targets: PhiTensor):
        """ Forward function.
        """
        raise NotImplementedError()

    def backward(self, outputs: PhiTensor, targets: PhiTensor):
        """Backward function.
        
        Parameters
        ----------
        outputs, targets : numpy.array 
            The arrays to compute the derivatives of them.
    
        Returns
        -------
        numpy.array 
            An array of derivative.
        """
        raise NotImplementedError()

    def __str__(self):
        return self.__class__.__name__


class BinaryCrossEntropy(Loss):
    def __init__(self, epsilon=1e-11):
        self.epsilon = epsilon

    def forward(self, outputs, targets):
        """Forward pass.

        .. math:: L = -t \\log(p) - (1 - t) \\log(1 - p)

        Parameters
        ----------
        outputs : numpy.array
            Predictions in (0, 1), such as sigmoidal output of a neural network.
        targets : numpy.array
            Targets in [0, 1], such as ground truth labels.
        """
        outputs = outputs.clip(self.epsilon, 1 - self.epsilon)
        log_loss = targets * dp_log(outputs) + ((targets * -1) + 1) * dp_log((outputs * -1) + 1)
        log_loss = log_loss.sum(axis=1) * -1
        return log_loss.mean()

    def backward(self, outputs: PhiTensor, targets: PhiTensor):
        """Backward pass.
        Parameters
        ----------
        outputs : numpy.array
            Predictions in (0, 1), such as sigmoidal output of a neural network.
        targets : numpy.array
            Targets in [0, 1], such as ground truth labels.
        """
        outputs = outputs.clip(self.epsilon, 1 - self.epsilon)
        divisor = dp_maximum(outputs * ((outputs * -1) + 1), self.epsilon)
        return (outputs - targets) * (1.0 / divisor)
