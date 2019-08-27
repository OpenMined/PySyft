import torch
from syft.workers import BaseWorker
from syft.frameworks.torch.linalg.operations import inv_sym
from syft.generic.pointers.pointer_tensor import PointerTensor
from typing import List


class BloomRegressor:
    def __init__(self, crypto_provider: BaseWorker, hbc_worker: BaseWorker, fit_intercept=True):
        self.crypto_provider = crypto_provider
        self.hbc_worker = hbc_worker
        self.fit_intercept = fit_intercept

    def fit(self, X_ptrs: List[torch.Tensor], y_ptrs: List[torch.Tensor]):

        # Checking if the pointers are as expected
        self._check_ptrs(X_ptrs, y_ptrs)

        self.workers = self._get_workers(X_ptrs)
        self._set_total_sample_size(X_ptrs)

        # Computing aggregated pairwise dot products remotelly
        XX_ptrs, Xy_ptrs, yy_ptrs = self._remote_dot_products(X_ptrs, y_ptrs)

    @staticmethod
    def _get_workers(ptrs):
        workers = []
        for ptr in ptrs:
            workers.append(ptr.child.location)
        return set(workers)

    @staticmethod
    def _check_ptrs(X_ptrs, y_ptrs):
        x_size, y_size = 0, 0
        for x, y in zip(X_ptrs, y_ptrs):
            if not (x.has_child and y.has_child):
                raise TypeError(
                    "Some tensors are not wrapped, please provide a wrapped Pointer Tensor"
                )
            if not (isinstance(x.child, PointerTensor) and isinstance(y.child, PointerTensor)):
                raise TypeError(
                    "Some tensors are not pointers, please provided a wrapped Pointer Tensor"
                )
            if not x.child.location == y.child.location:
                raise RuntimeError("Some pairs (X, y) are not located in the same worker")
            x_size += x.shape[0]
            y_size += y.shape[0]
            if x_size != y_size:
                raise ValueError("Some pairs (X, y) do not have the same number of samples")

    @staticmethod
    def _remote_dot_products(X_ptrs, y_ptrs):
        XX_ptrs = []
        Xy_ptrs = []
        yy_ptrs = []
        for x, y in zip(X_ptrs, y_ptrs):
            XX_ptrs.append(x.t() @ x)
            Xy_ptrs.append(x.t() @ y)
            yy_ptrs.append(y.t() @ y)

        return XX_ptrs, Xy_ptrs, yy_ptrs

    def _set_total_sample_size(self, X_ptrs):
        total_size = 0
        for x in X_ptrs:
            total_size += x.shape[0]
        self.total_size = total_size
