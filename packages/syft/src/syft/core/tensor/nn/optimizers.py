# stdlib
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

# third party
import numpy as np
from numpy.typing import NDArray

# relative
from ...common.serde.serializable import serializable
from ..autodp.gamma_tensor import GammaTensor
from ..autodp.phi_tensor import PhiTensor
from .layers.base import Layer
from .utils import dp_maximum


@serializable(recursive_serde=True)
class Optimizer:
    """Abstract optimizer base class.

    Parameters
    ----------
    clip : float
        If smaller than 0, do not apply parameter clip.
    lr : float
        The learning rate controlling the size of update steps
    decay : float
        Decay parameter for the moving average. Must lie in [0, 1) where
        lower numbers means a shorter “memory”.
    lr_min : float
        When adapting step rates, do not move below this value. Default is 0.
    lr_max : float
        When adapting step rates, do not move above this value. Default is inf.
    """

    __attr_allowlist__: Tuple[str, ...] = (
        "lr",
        "clip",
        "decay",
        "lr_min",
        "lr_max",
        "iterations",
    )

    def __init__(
        self,
        lr: float = 0.001,
        clip: int = -1,
        decay: float = 0.0,
        lr_min: float = 0.0,
        lr_max: float = np.inf,
        *args: Tuple,
        **kwargs: Dict,
    ):
        self.lr = lr
        self.clip = clip
        self.decay = decay
        self.lr_min = lr_min
        self.lr_max = lr_max

        self.iterations: int = 0

    def update(self, layers: List[Layer]) -> None:
        self.iterations += 1

        self.lr *= 1.0 / 1 + self.decay * self.iterations
        self.lr = np.clip(self.lr, self.lr_min, self.lr_max)

    def __str__(self) -> str:
        return self.__class__.__name__


@serializable(recursive_serde=True)
class Adamax(Optimizer):
    """
    Parameters
    ----------
    beta1 : float
        Exponential decay rate for the first moment estimates.
    beta2 : float
        Exponential decay rate for the second moment estimates.
    epsilon : float
        Constant for numerical stability.
    References
    ----------
    .. [1] Kingma, Diederik, and Jimmy Ba (2014):
           Adam: A Method for Stochastic Optimization.
           arXiv preprint arXiv:1412.6980.
    """

    __attr_allowlist__ = (
        "beta1",
        "beta2",
        "epsilon",
        "ms",
        "vs",
        "lr",
        "iterations",
    )

    def __init__(
        self,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        *args: Tuple,
        **kwargs: Dict,
    ):
        super(Adamax, self).__init__(args=args, kwargs=kwargs)

        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.ms: Optional[List[NDArray]] = None
        self.vs: Optional[List[NDArray]] = None

    def update(self, layers: List[Layer]) -> None:

        # init
        self.iterations += 1
        a_t = self.lr / (1 - np.power(self.beta1, self.iterations))

        params: List[Union[np.ndarray, PhiTensor, GammaTensor]] = []
        grads: List[Union[np.ndarray, PhiTensor, GammaTensor]] = []
        for layer in layers:
            params += layer.params
            grads += layer.grads

        if self.ms is None:
            self.ms = [np.zeros(p.shape) for p in params]
        if self.vs is None:
            self.vs = [np.zeros(p.shape) for p in params]

        idx = 0
        for layer in layers:
            new_params: List[Layer] = []

            for p, g in zip(layer.params, layer.grads):
                m = self.ms[idx]
                v = self.vs[idx]
                m = g * (1 - self.beta1) + m * self.beta1
                v = dp_maximum(x=g.abs(), y=(v * self.beta2))  # type: ignore
                p = (m * (-1.0 / (v + self.epsilon)) * a_t) + p
                new_params.append(p)

                self.ms[idx] = m
                self.vs[idx] = v
                idx += 1
            if new_params:
                layer.params = new_params  # type: ignore
