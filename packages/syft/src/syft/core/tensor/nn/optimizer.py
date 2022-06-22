from typing import Optional, Callable

from torch.optim import Optimizer


class AdaMax(Optimizer):

    def __init__(self, params, lr=2e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):

        if lr < 0:
            raise Exception("Learning rate must be greater than 0. Otherwise you're not learning, you're forgetting.")
        if not 0 <= betas[0] < 1 and 0 <= betas[1] < 1:
            raise Exception("Beta parameters should be in the range of [0, 1).")
        if not 0 <= eps:
            raise Exception("Epsilon should be >= 0")


        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(AdaMax, self).__init__(params=params, default=defaults)

    def step(self, closure: Optional[Callable[[], float]]=...) -> Optional[float]:
        pass


