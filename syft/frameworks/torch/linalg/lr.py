import random
from typing import List

import torch

from syft.workers.base import BaseWorker
from syft.frameworks.torch.linalg.operations import inv_sym
from syft.frameworks.torch.linalg.operations import qr
from syft.generic.pointers.pointer_tensor import PointerTensor

from syft.exceptions import DependencyError

try:
    from scipy.stats import t
except ImportError:
    raise DependencyError("scipy", "scipy")


class EncryptedLinearRegression:
    """
    Multi-Party Linear Regressor based on Jonathan Bloom's algorithm.
    It performs linear regression using Secure Multi-Party Computation.
    While the training is performed in SMPC, the final regression coefficients
    are public at the end and predictions are made in clear on local or pointer
    Tensors.

    Reference: Section 2 of https://arxiv.org/abs/1901.09531

    Args:
        crypto_provider: a BaseWorker providing crypto elements for
            AdditiveSharingTensors (used for SMPC) such as Beaver triples
        hbc_worker: The "Honest but Curious" BaseWorker. SMPC operations in PySyft
            use SecureNN protocols, which are based on 3-party computations. In order to
            apply it for more than 3 parties, we need a "Honest but Curious" worker.
            To perform the Encrypted Linear Regression, the algorithm chooses randomly one of the
            workers in the pool and secret share all tensors with the chosen worker,
            the crypto provider and the "Honest but Curious" worker. Its main role
            is to avoid collusion between two workers in the pool if the algorithm
            secred shared the tensors with two randomly chosen workers and the
            crypto provider. The "Honest but Curious" worker is essentially a
            legitimate participant in a communication protocol who will not deviate
            from the defined protocol but will attempt to learn all possible
            information from legitimately received messages.
        precision_fractional: precision chosen for FixedPrecisionTensors
        fit_intercept:  whether to calculate the intercept for this model. If set
            to False, no intercept will be used in calculations (e.g. data is
            expected to be already centered)

    Attributes:
        coef: torch.Tensor of shape (n_features, ). Estimated coefficients for
            the linear regression problem.
        intercept: torch.Tensor of shape (1, ) if fit_intercept is set to True,
            None otherwise. Estimated intercept for the linear regression.
        pvalue_coef: numpy.array of shape (n_features, ). Two-sided p-value for a
            hypothesis test whose null hypothesis is that the each coeff is zero.
        pvalue_intercept: numpy.array of shape (1, ) if fit_intercept is set to True,
            None otherwise. Two-sided p-value for a hypothesis test whose null
            hypothesis is that the intercept is zero.

    """

    def __init__(
        self,
        crypto_provider: BaseWorker,
        hbc_worker: BaseWorker,
        precision_fractional: int = 6,
        fit_intercept: bool = True,
        protocol: str = "snn",
    ):

        self.crypto_provider = crypto_provider
        self.hbc_worker = hbc_worker
        self.precision_fractional = precision_fractional
        self.fit_intercept = fit_intercept
        self.protocol = protocol

    def fit(self, X_ptrs: List[torch.Tensor], y_ptrs: List[torch.Tensor]):
        """
        Fits the linear model using Secured Multi-Party Linear Regression.
        The final results (i.e. coefficients and p-values) will be public.
        """

        # Checking if the pointers are as expected
        self._check_ptrs(X_ptrs, y_ptrs)

        # Check if each y is a 2-dim or 1-dim tensor, unsqueeze it if it's 1-dim
        for i, y in enumerate(y_ptrs):
            if len(y.shape) < 2:
                y_ptrs[i] = y.unsqueeze(1)

        if self.fit_intercept:
            X_ptrs = self._add_intercept(X_ptrs)
            self._dgf -= 1

        self.workers = self._get_workers(X_ptrs)

        # Computing aggregated pairwise dot products remotelly
        XX_ptrs, Xy_ptrs, yy_ptrs = self._remote_dot_products(X_ptrs, y_ptrs)

        # Secred share tensors between hbc_worker, crypto_provider and a random worker
        # and compute aggregates. It corresponds to the Combine stage of Bloom's algorithm
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

        XX_shared = XX_shared / self.total_size
        XX_inv_shared = inv_sym(XX_shared)

        # Compute shared coefficients
        coef_shared = XX_inv_shared @ Xy_shared

        sigma2_shared = yy_shared * self.total_size - coef_shared.t() @ XX_shared @ coef_shared
        sigma2_shared = sigma2_shared / self._dgf

        var_diag_shared = torch.diag(XX_inv_shared) * sigma2_shared

        # Store results locally and resize by dividing by total_size
        self.coef = coef_shared.get().float_precision() / self.total_size
        self.coef = self.coef.squeeze()
        self.se_coef = torch.sqrt(var_diag_shared.get().float_precision()) / self.total_size
        self.se_coef = self.se_coef.squeeze()

        self.sigma = torch.sqrt(sigma2_shared.get().float_precision().squeeze() / self.total_size)

        if self.fit_intercept:
            self.intercept = self.coef[0]
            self.coef = self.coef[1:]
            self.se_intercept = self.se_coef[0]
            self.se_coef = self.se_coef[1:]
        else:
            self.intercept = None
            self.se_intercept = None

        self._compute_pvalues()

        return self

    def predict(self, X: torch.Tensor):
        """
        Performs predicion of linear model on X, which can be a local torch.Tensor
        or a wrapped PointerTensor. The result will be either a local torch.Tensor
        or a wrapped PointerTensor, depending on the nature of X.
        """
        coef = self.coef.copy()
        intercept = self.intercept.copy() if self.fit_intercept else None

        # Send coef and intercept to remote worker if X is a pointer
        if X.has_child() and isinstance(X.child, PointerTensor):
            coef = coef.send(X.child.location)
            if self.fit_intercept:
                intercept = intercept.send(X.child.location)

        y = X @ coef.unsqueeze(1)
        if self.fit_intercept:
            y += intercept
        return y.squeeze()

    def summarize(self):
        """
        Prints a summary of the coefficients and its statistics.
        This method should be called only after training of the model.
        """
        print("=" * 52)
        print(" " * 11 + "SMPC Linear Regression Results")
        print("=" * 52)
        print(" " * 17 + "value" + " " * 9 + "stderr" + " " * 8 + "p-value")
        print("-" * 52)
        for i, cf in enumerate(self.coef):
            print(
                "coef" + f"{i + 1:<3d}",
                f"{cf:>14.4f}",
                f"{self.se_coef[i]:>14.4f}",
                f"{self.pvalue_coef[i]:>14.4f}",
            )
        if self.fit_intercept:
            print(
                "intercept",
                f"{self.intercept:>12.4f}",
                f"{self.se_intercept:>14.4f}",
                f"{self.pvalue_intercept:>14.4f}",
            )
        print("-" * 52)

    def _check_ptrs(self, X_ptrs, y_ptrs):
        """
        Method that check if the lists of pointers corresponding to the explanatory and
        explained variables have their elements as expected.
        It also computes parallelly some Regressor's attributes such as number of features and
        total sample size.
        """
        # Set number of features
        self.n_features = X_ptrs[0].shape[1]

        x_size, y_size = 0, 0
        for x, y in zip(X_ptrs, y_ptrs):
            # Check wrapper
            if not (x.has_child() and y.has_child()):
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
        X_ptrs_new = []
        for i, x in enumerate(X_ptrs):
            ones = torch.ones_like(x[:, :1])
            x = torch.cat((ones, x), 1)
            X_ptrs_new.append(x)
        return X_ptrs_new

    @staticmethod
    def _get_workers(ptrs):
        """
        Method that returns the pool of workers in a tuple
        """
        workers = set()
        for ptr in ptrs:
            workers.add(ptr.child.location)
        return tuple(workers)

    @staticmethod
    def _remote_dot_products(X_ptrs, y_ptrs):
        """
        This method computes the aggregated dot-products remotely. It corresponds
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
                self.workers[worker_idx],
                self.hbc_worker,
                crypto_provider=self.crypto_provider,
                protocol=self.protocol,
            ).get()
            shared_tensors.append(shared_tensor)
        return shared_tensors

    def _compute_pvalues(self):
        """
        Compute p-values of coefficients (and intercept if fit_intercept==True)
        """
        tstat_coef = self.coef / self.se_coef
        self.pvalue_coef = 2 * t.cdf(-abs(tstat_coef), self._dgf)

        if self.fit_intercept:
            tstat_intercept = self.intercept / self.se_intercept
            self.pvalue_intercept = 2 * t.cdf(-abs(tstat_intercept), self._dgf)
        else:
            self.pvalue_intercept = None


class DASH:
    """
    Distributed Association Scan Hammer (DASH) algorithm based on Jonathan Bloom's algorithm.
    It uses Secured Multi-Party Computation at combine phase.
    While the training is performed in SMPC, the final regression coefficients
    are public at the end.

    Reference: Section 2 of https://arxiv.org/abs/1901.09531

    Args:
        crypto_provider: a BaseWorker providing crypto elements for ASTs such as
            Beaver triples
        hbc_worker: The "Honest but Curious" BaseWorker. SMPC operations in PySyft
            use SecureNN protocols, which are based on 3-party computations. In order to
            apply it for more than 3 parties, we need a "Honest but Curious" worker.
            To perform the DASH algorithm, we choose randomly one of the workers
            in the pool and secret share all tensors with the chosen worker,the crypto
            provider and the "Honest but Curious" worker. Its main role is to avoid
            collusion between two workers in the pool if the algorithm secred shared
            the tensors with two randomly chosen workers and the crypto provider.
            The "Honest but Curious" worker is essentially a legitimate participant
            in a communication protocol who will not deviate from the defined protocol
            but will attempt to learn all possible information from legitimately
            received messages.
        precision_fractional: precision chosen for FixedPrecisionTensors
        protocol: the crypto protocol used for private comparison

    Attributes:
        coef: torch.Tensor of shape (n_features, ). Estimated coefficients for
            DASH algorithm.
        pvalue: numpy.array of shape (n_features, ). Two-sided p-value for a
            hypothesis test whose null hypothesis is that the each coeff is zero.

    """

    def __init__(
        self,
        crypto_provider: BaseWorker,
        hbc_worker: BaseWorker,
        precision_fractional: int = 6,
        protocol: str = "snn",
    ):

        self.crypto_provider = crypto_provider
        self.hbc_worker = hbc_worker
        self.precision_fractional = precision_fractional
        self.protocol = protocol

    def fit(
        self, X_ptrs: List[torch.Tensor], C_ptrs: List[torch.Tensor], y_ptrs: List[torch.Tensor]
    ):

        # Checking if the pointers are as expected
        self._check_ptrs(X_ptrs, C_ptrs, y_ptrs)

        # Check if each y is a 2-dim or 1-dim tensor, unsqueeze it if it's 1-dim
        for i, y in enumerate(y_ptrs):
            if len(y.shape) < 2:
                y_ptrs[i] = y.unsqueeze(1)

        self.workers = self._get_workers(X_ptrs)

        # Computing aggregated pairwise dot products remotely
        XX_ptrs, Xy_ptrs, yy_ptrs, CX_ptrs, Cy_ptrs = self._remote_dot_products(
            X_ptrs, C_ptrs, y_ptrs
        )

        # Compute remote QR decompositions
        R_ptrs = self._remote_qr(C_ptrs)

        # Secred share tensors between hbc_worker, crypto_provider and a random worker
        # and compute aggregates. It corresponds to the Combine stage of DASH's algorithm
        idx = random.randint(0, len(self.workers) - 1)
        XX_shared = sum(self._share_ptrs(XX_ptrs, idx))
        Xy_shared = sum(self._share_ptrs(Xy_ptrs, idx))
        yy_shared = sum(self._share_ptrs(yy_ptrs, idx))
        CX_shared = sum(self._share_ptrs(CX_ptrs, idx))
        Cy_shared = sum(self._share_ptrs(Cy_ptrs, idx))
        R_cat_shared = torch.cat(self._share_ptrs(R_ptrs, idx), dim=0)

        # QR decomposition of R_cat_shared
        _, R_shared = qr(R_cat_shared, norm_factor=self.total_size ** (1 / 2))

        # Compute inverse of upper matrix
        R_shared_inv = self._inv_upper(R_shared)

        Qy = R_shared_inv.t() @ Cy_shared
        QX = R_shared_inv.t() @ CX_shared

        denominator = XX_shared - (QX ** 2).sum(dim=0)
        # Need the line below to perform inverse of a number in MPC
        inv_denominator = ((0 * denominator + 1) / denominator).squeeze()

        coef_shared = (Xy_shared - QX.t() @ Qy).squeeze() * inv_denominator

        sigma2_shared = (
            (yy_shared - Qy.t() @ Qy).squeeze() * inv_denominator - coef_shared ** 2
        ) / self._dgf

        self.coef = coef_shared.get().float_precision()
        self.sigma2 = sigma2_shared.get().float_precision()
        self.se = self.sigma2 ** (1 / 2)

        self._compute_pvalues()

    def get_coeff(self):
        return self.coef

    def get_standard_errors(self):
        return self.se

    def get_p_values(self):
        return self.pvalue

    def _check_ptrs(self, X_ptrs, C_ptrs, y_ptrs):
        """
        Method that check if the lists of pointers corresponding to the response vector,
        transient covariate vectors and independent permanent covariate vectors have
        their elements as expected. It also computes parallelly some Regressor's
        attributes such as degrees of freedom and total sample size.
        """
        # Set number of features
        self.n_features = X_ptrs[0].shape[1]
        # Set number of permanent covariate features
        self.n_permanent = C_ptrs[0].shape[1]

        x_size, c_size, y_size = 0, 0, 0
        for x, c, y in zip(X_ptrs, C_ptrs, y_ptrs):
            # Check wrappers and if x, c and y are pointers
            if not all(
                map(lambda t: t.has_child() and isinstance(t.child, PointerTensor), (x, c, y))
            ):
                raise TypeError(
                    "Some tensors are not pointers or are not wrapped, please provided a wrapped "
                    "Pointer Tensor"
                )

            # Check if both are in the same worker
            if not (x.child.location == c.child.location and x.child.location == y.child.location):
                raise RuntimeError("Some tuples (X, C, y) are not located in the same worker")

            # Check if they have the same size
            x_size += x.shape[0]
            c_size += c.shape[0]
            y_size += y.shape[0]
            if x_size != c_size or x_size != y_size:
                raise ValueError("Some tuples (X, C, y) do not have the same number of samples")

            # Check if all tensors have the same number of features
            if x.shape[1] != self.n_features:
                raise ValueError(
                    "Transient covariate vectors do not have the same number of features"
                )

            if c.shape[1] != self.n_permanent:
                raise ValueError(
                    "Permanent covariate vectors do not have the same number of features"
                )

        # Set total size
        self.total_size = x_size

        # Set degrees of freedom
        self._dgf = self.total_size - self.n_permanent - 1

    @staticmethod
    def _get_workers(ptrs):
        """
        Method that returns the pool of workers in a tuple
        """
        workers = set()
        for ptr in ptrs:
            workers.add(ptr.child.location)
        return tuple(workers)

    @staticmethod
    def _remote_dot_products(X_ptrs, C_ptrs, y_ptrs):
        """
        This method computes the aggregated dot-products remotely. It corresponds
        to the Compression stage (or Compression within) of DASH algorithm
        """
        XX_ptrs = []
        Xy_ptrs = []
        yy_ptrs = []
        CX_ptrs = []
        Cy_ptrs = []
        for x, c, y in zip(X_ptrs, C_ptrs, y_ptrs):
            XX_ptrs.append((x.t() @ x).sum(dim=0))
            Xy_ptrs.append(x.t() @ y)
            yy_ptrs.append(y.t() @ y)
            CX_ptrs.append(c.t() @ x)
            Cy_ptrs.append(c.t() @ y)

        return XX_ptrs, Xy_ptrs, yy_ptrs, CX_ptrs, Cy_ptrs

    @staticmethod
    def _remote_qr(C_ptrs):
        """
        Performs the QR decompositions of permanent covariate matrices remotely.
        It returns a list with the upper right matrices located in each worker
        """
        R_ptrs = []
        for c in C_ptrs:
            _, r = qr(c)
            R_ptrs.append(r)
        return R_ptrs

    @staticmethod
    def _inv_upper(R):
        """
        Performs the inversion of a right upper matrix (2-dim tensor) in MPC by
        solving the linear equation R * R_inv = I with backward substitution.
        """
        R_inv = torch.zeros_like(R)
        N = R.shape[0]

        # Identity Matrix
        I = torch.zeros_like(R)
        for i in range(N):
            I[i, i] += 1

        for i in range(N - 1, -1, -1):
            if i == N - 1:
                R_inv[i, :] = I[i, :] / R[i, i]
            else:
                R_inv[i, :] = I[i, :] - (R[i : i + 1, (i + 1) : N].t() * R_inv[(i + 1) : N, :]).sum(
                    dim=0
                )
                R_inv[i, :] = R_inv[i, :] / R[i, i]

        return R_inv

    def _share_ptrs(self, ptrs, worker_idx):
        """
        Method that secret share a list of remote tensors between a worker of
        the pool and the 'honest but curious' worker, using a crypto_provider worker
        """
        shared_tensors = []
        for ptr in ptrs:
            fpt_tensor = ptr.fix_precision(precision_fractional=self.precision_fractional)
            shared_tensor = fpt_tensor.share(
                self.workers[worker_idx],
                self.hbc_worker,
                crypto_provider=self.crypto_provider,
                protocol=self.protocol,
            ).get()
            shared_tensors.append(shared_tensor)
        return shared_tensors

    def _compute_pvalues(self):
        """
        Compute p-values of coefficients
        """
        tstat = self.coef / self.se
        self.pvalue = 2 * t.cdf(-abs(tstat), self._dgf)
