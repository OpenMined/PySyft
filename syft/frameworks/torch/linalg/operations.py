import torch as th
import syft as sy
from typing import List
from syft.workers import BaseWorker


def inv_sym(a):  # , workers: List, crypto_provider: BaseWorker):
    # TODO
    pass


def ldl(a):
    n = a.shape[0]
    l = th.zeros_like(a)
    d = th.diag(l).copy()
    inv_d = d.copy()

    for i in range(n):
        d[i] = a[i, i] - (l[i, :i] ** 2 * d[:i]).sum()
        inv_d[i] = 1.0 / d[i]
        for j in range(i, n):
            l[j, i] = (a[j, i] - (l[j, :i] * l[i, :i] * d[:i]).sum()) * inv_d[i]

    return l, d, inv_d


def qr_remote(a):
    # TODO
    pass


def qr_mpc(a, workers: List, crypto_provider: BaseWorker):
    # TODO
    pass
