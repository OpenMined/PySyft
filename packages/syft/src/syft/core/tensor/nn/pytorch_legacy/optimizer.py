# stdlib
from typing import Callable
from typing import Optional

# third party
import torch
from torch.optim import Optimizer

# relative
from ..autodp.phi_tensor import PhiTensor


class AdaMax(Optimizer):
    def __init__(self, params, lr=2e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):

        if lr < 0:
            raise Exception(
                "Learning rate must be greater than 0. Otherwise you're not learning, you're forgetting."
            )
        if not 0 <= betas[0] < 1 and 0 <= betas[1] < 1:
            raise Exception("Beta parameters should be in the range of [0, 1).")
        if not 0 <= eps:
            raise Exception("Epsilon should be >= 0")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(AdaMax, self).__init__(params=params, default=defaults)

    def step(self, closure: Optional[Callable[[], float]] = None) -> PhiTensor:
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise Exception("AdaMax can't handle sparse gradients")
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_inf"] = torch.zeros_like(p.data)

                exp_avg, exp_inf = state["exp_avg"], state["exp_inf"]
                beta1, beta2 = group["betas"]
                eps = group["eps"]

                state["step"] += 1

                if group["weight_decay"] != 0:
                    grad = grad.add(group["weight_decay"], p.data)

                # Update biased first moment estimate.
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                # Update the exponentially weighted infinity norm.
                norm_buf = torch.cat(
                    [
                        exp_inf.mul_(beta2).unsqueeze(0),
                        grad.abs().add_(eps).unsqueeze_(0),
                    ],
                    0,
                )
                torch.max(
                    norm_buf, 0, keepdim=False, out=(exp_inf, exp_inf.new().long())
                )

                bias_correction = 1 - beta1 ** state["step"]
                clr = group["lr"] / bias_correction

                p.data.addcdiv_(-clr, exp_avg, exp_inf)

            return loss
