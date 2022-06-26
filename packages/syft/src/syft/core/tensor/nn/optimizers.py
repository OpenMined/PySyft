# third party
import numpy as np

# relative
from .utils import dp_maximum


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

    def __init__(self, lr=0.001, clip=-1, decay=0., lr_min=0., lr_max=np.inf):
        self.lr = lr
        self.clip = clip
        self.decay = decay
        self.lr_min = lr_min
        self.lr_max = lr_max

        self.iterations = 0

    def update(self, params, grads):
        self.iterations += 1

        self.lr *= (1. / 1 + self.decay * self.iterations)
        self.lr = np.clip(self.lr, self.lr_min, self.lr_max)

    def __str__(self):
        return self.__class__.__name__


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

    def __init__(self, beta1=0.9, beta2=0.999, epsilon=1e-8, *args, **kwargs):
        super(Adamax, self).__init__(*args, **kwargs)

        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.ms = None
        self.vs = None

    def update(self, params, grads):
        # init
        self.iterations += 1
        a_t = self.lr / (1 - np.power(self.beta1, self.iterations))
        if self.ms is None:
            self.ms = [p.zeros_like() for p in params]
        if self.vs is None:
            self.vs = [p.zeros_like() for p in params]

        # update parameters
        for i, (m, v, p, g) in enumerate(zip(self.ms, self.vs, params, grads)):
            m = m * self.beta1 + g * (1 - self.beta1)
            v = dp_maximum(v * self.beta2, g.abs())
            p = p - (m * (1.0 / (v + self.epsilon)) * a_t)

            self.ms[i] = m
            self.vs[i] = v
