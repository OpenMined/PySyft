import syft as sy
import torch as th

n = 32  # Could be less if you expect data to be < 2**32


def power(x_sh):
    assert isinstance(x_sh, sy.AdditiveSharingTensor)

    alpha = th.zeros(*x_sh.shape)

    for i in range(n):
        beyond = ((x_sh - 2 ** i) < 0).get()
        alpha = i * beyond * (alpha == 0) + alpha * (alpha != 0)

        if (beyond == 1).all():
            break

    return alpha
