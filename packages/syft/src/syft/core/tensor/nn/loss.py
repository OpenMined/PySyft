# third party

# stdlib
from typing import Tuple
from typing import Union

# third party
import numpy as np

# relative
from ...common.serde.serializable import serializable
from ..autodp.gamma_tensor import GammaTensor
from ..autodp.phi_tensor import PhiTensor
from .utils import dp_log


@serializable(recursive_serde=True)
class Loss(object):
    """An objective function (or loss function, or optimization score
    function) is one of the two parameters required to compile a model.

    """

    __attr_allowlist__: Tuple[str, ...] = ()

    def forward(
        self,
        outputs: Union[PhiTensor, GammaTensor],
        targets: Union[PhiTensor, GammaTensor],
    ) -> Union[PhiTensor, GammaTensor]:
        """Forward function."""
        raise NotImplementedError()

    def backward(
        self,
        outputs: Union[PhiTensor, GammaTensor],
        targets: Union[PhiTensor, GammaTensor],
    ) -> Union[PhiTensor, GammaTensor]:
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

    def __str__(self) -> str:
        return self.__class__.__name__


@serializable(recursive_serde=True)
class BinaryCrossEntropy(Loss):
    __attr_allowlist__ = ("epsilon",)

    def __init__(self, epsilon: float = 1e-11):
        self.epsilon = epsilon

    def forward(
        self,
        outputs: Union[PhiTensor, GammaTensor],
        targets: Union[PhiTensor, GammaTensor],
    ) -> Union[PhiTensor, GammaTensor]:
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
        log_loss = targets * dp_log(outputs) + ((targets * -1) + 1) * dp_log(
            (outputs * -1) + 1
        )
        log_loss = log_loss.sum(axis=1) * -1
        return log_loss.mean(axis=0)

    def backward(
        self,
        outputs: Union[PhiTensor, GammaTensor],
        targets: Union[PhiTensor, GammaTensor],
    ) -> Union[PhiTensor, GammaTensor]:
        """Backward pass.
        Parameters
        ----------
        outputs : numpy.array
            Predictions in (0, 1), such as sigmoidal output of a neural network.
        targets : numpy.array
            Targets in [0, 1], such as ground truth labels.
        """
        outputs = outputs.clip(self.epsilon, 1 - self.epsilon)
        divisor = outputs * ((outputs * -1) + 1)
        divisor = np.maximum(divisor.child, self.epsilon)
        return (outputs - targets) * (1.0 / divisor)
