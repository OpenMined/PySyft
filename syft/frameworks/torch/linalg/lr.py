import torch
from syft.workers import BaseWorker
from syft.frameworks.torch.linalg.operations import inv_sym
from syft.generic.pointers.pointer_tensor import PointerTensor
from typing import List
import random


class BloomRegressor:
    """
    Multi-Party Linear Regressor based on Jonathan Bloom's algorithm.
    It performs linear regression using Secured Multi-Party Computation

    Reference: Section 2 of https://arxiv.org/abs/1901.09531

    Args:
        cryto_provider: a BaseWorker providing crypto elements for ASTs such as
            Beaver triples
        hbc_worker: The "Honest but Curious" BaseWorker
        precision_fractional: precision chosen for FixedPrecisionTensors
        fit_intercept:  whether to calculate the intercept for this model. If set
            to False, no intercept will be used in calculations (e.g. data is
            expected to be already centered)

    """

    def __init__(
        self,
        crypto_provider: BaseWorker,
        hbc_worker: BaseWorker,
        precision_fractional: int = 6,
        fit_intercept=True,
    ):

        self.crypto_provider = crypto_provider
        self.hbc_worker = hbc_worker
        self.precision_fractional = precision_fractional
        self.fit_intercept = fit_intercept

    def fit(self, X_ptrs: List[torch.Tensor], y_ptrs: List[torch.Tensor]):

        # Checking if the pointers are as expected
        self._check_ptrs(X_ptrs, y_ptrs)

        if self.fit_intercept:
            self._add_intercept(X_ptrs)
            self._dgf -= 1

        self.workers = self._get_workers(X_ptrs)

        # Computing aggregated pairwise dot products remotelly
        XX_ptrs, Xy_ptrs, yy_ptrs = self._remote_dot_products(X_ptrs, y_ptrs)

        # Secred share tensors between hbc_worker, crypto_provider and a random worker
        # and compute agregates. It corresponds to the Combine stage of Bloom's algorithm
        idx = random.randint(0, len(self.workers) - 1)
        XX_shared = sum(self._share_ptrs(XX_ptrs, idx))
        Xy_shared = sum(self._share_ptrs(Xy_ptrs, idx))
        yy_shared = sum(self._share_ptrs(yy_ptrs, idx))

        ##################### Compute inverse of Gram Matrix ###########################
        # We need to normalize it by dividing the Gram matrix by the total sample size
        # because matrix inversion in MPC is not precise for large numbers, which is the
        # case of the Gram matrix when total_size is large. We only resize back the
        # values we are interested in (i.e. the coefficients and std errors) locally at
        # the end in order to make sure the subsequent computations are still precise

        XX_inv_shared = inv_sym(XX_shared / self.total_size)

        # Compute shared coefficients
        coef_shared = XX_inv_shared @ Xy_shared

        sigma2_shared = yy_shared - coef_shared.t() @ XX_shared @ coef_shared
        sigma2_shared = sigma2_shared / self._dgf

        var_diag_shared = torch.diag(XX_inv_shared) * sigma2_shared

        # Store results locally and resize by dividing by total_size
        self.coef = coef_shared.get().float_precision() / self.total_size
        self.se_coef = torch.sqrt(var_diag_shared.get().float_precision() / self.total_size)

        if self.fit_intercept:
            self.intercept = self.coef[0]
            self.coef = self.coef[1:]
            self.se_intercpt = self.se_coef[0]
            self.se_coef = self.se_coef[1:]
        else:
            self.intercept = None
            self.se_intercept = None

        return self

    def _check_ptrs(self, X_ptrs, y_ptrs):
        """
        Method that check if the lists of pointers corresponding to the explanatory and
        explained variables has their elements as expected.
        It also computes parallelly  some Regressor's attributes such as number of features and
        total sample size.
        """

        # Set number of features
        self.n_features = X_ptrs[0].shape[1]

        x_size, y_size = 0, 0
        for x, y in zip(X_ptrs, y_ptrs):
            # Check wrapper
            if not (x.has_child and y.has_child):
                raise TypeError(
                    "Some tensors are not wrapped, please provide a wrapped Pointer Tensor"
                )

            # Check if x and y are pointers
            if not (isinstance(x.child, PointerTensor) and isinstance(y.child, PointerTensor)):
                raise TypeError(
                    "Some tensors are not pointers, please provided a wrapped Pointer Tensor"
                )

            # Check if both are in the same worker
            if not x.child.location == y.child.location:
                raise RuntimeError("Some pairs (X, y) are not located in the same worker")

            # Check if they have the same size
            x_size += x.shape[0]
            y_size += y.shape[0]
            if x_size != y_size:
                raise ValueError("Some pairs (X, y) do not have the same number of samples")

            # Check if all tensors have the same number of features
            if x.shape[1] != self.n_features:
                raise ValueError("Tensors do not have the same number of features")

        # Set total size
        self.total_size = x_size

        # Set degrees of freedom
        self._dgf = self.total_size - self.n_features

    @staticmethod
    def _add_intercept(X_ptrs):
        """
        Adds a column-vector of 1's at the beginning of the tensors X_ptrs
        """
        for i, x in enumerate(X_ptrs):
            ones = torch.ones_like(x[:, :1])
            x = torch.cat((ones, x), 1)
            X_ptrs[i] = x

    @staticmethod
    def _get_workers(ptrs):
        """
        Method that returns the pool of workers in a tuple
        """
        workers = []
        for ptr in ptrs:
            workers.append(ptr.child.location)
        return tuple(set(workers))

    @staticmethod
    def _remote_dot_products(X_ptrs, y_ptrs):
        """
        This method computes the aggregated dot-products remotelly. It corresponds
        to the Compression stage (or Compression within) of Bloom's algorithm
        """
        XX_ptrs = []
        Xy_ptrs = []
        yy_ptrs = []
        for x, y in zip(X_ptrs, y_ptrs):
            XX_ptrs.append(x.t() @ x)
            Xy_ptrs.append(x.t() @ y)
            yy_ptrs.append(y.t() @ y)

        return XX_ptrs, Xy_ptrs, yy_ptrs

    def _share_ptrs(self, ptrs, worker_idx):
        """
        Method that secret share a list of remote tensors between a worker of
        the pool and the 'honest but curious' worker, using a crypto_provider worker
        """
        shared_tensors = []
        for ptr in ptrs:
            fpt_tensor = ptr.fix_precision(precision_fractional=self.precision_fractional)
            shared_tensor = fpt_tensor.share(
                self.workers[worker_idx], self.hbc_worker, crypto_provider=self.crypto_provider
            ).get()
            shared_tensors.append(shared_tensor)
        return shared_tensors

    def summarize(self):
        # TODO
        pass

    def predict(self):
        # TODO
        pass
