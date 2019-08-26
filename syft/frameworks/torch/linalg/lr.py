import torch as th
from typing import List

import syft
from syft.workers import BaseWorker
from syft.frameworks.linalg.operations import inv_sym


class BloomRegressor:
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept

    def fit(self, X: List, y: List, crypto_provider: BaseWorker):
        pass
