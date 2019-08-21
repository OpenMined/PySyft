import torch as th
import syft as sy
from typing import List
from syft.workers import BaseWorker


def qr_mpc(a, workers: List, crypto_provider: BaseWorker):
    # a = a.float_precision().share(*workers, crypto_provider=crypto_provider)
    m, n = a.shape
