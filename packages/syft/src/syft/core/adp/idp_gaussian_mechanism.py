# CLEANUP NOTES:
# - remove unused comments
# - add documentation for each method
# - add comments inline explaining each piece
# - add a unit test for each method (at least)

# stdlib
from functools import lru_cache
from typing import Dict
from typing import List
from typing import Optional

# third party
from autodp import dp_bank
from autodp import fdp_bank
from autodp.autodp_core import Mechanism
from nacl.signing import VerifyKey
import numpy as np

# relative
from ..common.serde.serializable import serializable


# methods serialize/deserialize np.int64 number
# syft.serde seems to not support np.int64 serialization/deserialization
def numpy64tolist(value: np.int64) -> List:
    list_version = value.tolist()
    return list_version


def listtonumpy64(value: List) -> np.int64:
    return np.int64(value)


# returns the privacy budget spent by each entity
@lru_cache(maxsize=None)
def _individual_RDP_gaussian(
    sigma: float, value: float, L: float, alpha: float
) -> float:
    return (alpha * (L**2) * (value**2)) / (2 * (sigma**2))


def individual_RDP_gaussian(params: Dict, alpha: float) -> np.float64:
    """
    :param params:
        'sigma' --- is the normalized noise level: std divided by global L2 sensitivity
        'value' --- is the output of query on a data point
        'L' --- is the Lipschitz constant of query with respect to the output of query on a data point
    :param alpha: The order of the Renyi Divergence
    :return: Evaluation of the RDP's epsilon
    """
    sigma = params["sigma"]
    value = params["value"]
    L = params["L"]
    if sigma <= 0:
        raise Exception("Sigma should be above 0")
    if alpha < 0:
        raise Exception("Sigma should not be below 0")

    return _individual_RDP_gaussian(sigma=sigma, alpha=alpha, value=value, L=L)


# Example of a specific mechanism that inherits the Mechanism class
@serializable(recursive_serde=True)
class iDPGaussianMechanism(Mechanism):
    __attr_allowlist__ = [
        "name",
        "params",
        "entity_name",
        "fdp",
        "eps_pureDP",
        "delta0",
        "RDP_off",
        "approxDP_off",
        "fdp_off",
        "use_basic_rdp_to_approx_dp_conversion",
        "use_fdp_based_rdp_to_approx_dp_conversion",
        "user_key",
    ]

    # delta0 is a numpy.int64 number (not supported by syft.serde)
    __serde_overrides__ = {
        "delta0": [numpy64tolist, listtonumpy64],
    }

    def __init__(
        self,
        sigma: float,
        squared_l2_norm: float,
        squared_l2_norm_upper_bound: float,
        L: float,
        entity_name: str,
        name: str = "Gaussian",
        RDP_off: bool = False,
        approxDP_off: bool = False,
        fdp_off: bool = True,
        use_basic_rdp_to_approx_dp_conversion: bool = False,
        use_fdp_based_rdp_to_approx_dp_conversion: bool = False,
        user_key: Optional[VerifyKey] = None,
    ):

        # the sigma parameter is the std of the noise divide by the l2 sensitivity
        Mechanism.__init__(self)

        self.user_key = user_key

        self.name = name  # When composing
        self.params = {
            "sigma": float(sigma),
            "private_value": float(squared_l2_norm),
            "public_value": float(squared_l2_norm_upper_bound),
            "L": float(L),
        }  # This will be useful for the Calibrator

        self.entity_name = entity_name
        # TODO: should a generic unspecified mechanism have a name and a param dictionary?

        self.delta0 = 0
        if not RDP_off:
            # Tudor: i'll fix these
            new_rdp = lambda x: individual_RDP_gaussian(self.params, x)  # noqa: E731
            if use_fdp_based_rdp_to_approx_dp_conversion:
                # This setting is slightly more complex, which involves converting RDP to fDP,
                # then to eps-delta-DP via the duality
                self.propagate_updates(new_rdp, "RDP", fDP_based_conversion=True)
            elif use_basic_rdp_to_approx_dp_conversion:
                self.propagate_updates(new_rdp, "RDP", BBGHS_conversion=False)
            else:
                # This is the default setting with fast computation of RDP to approx-DP
                self.propagate_updates(new_rdp, "RDP")

        if not approxDP_off:  # Direct implementation of approxDP
            new_approxdp = lambda x: dp_bank.get_eps_ana_gaussian(  # noqa: E731
                sigma, x
            )
            self.propagate_updates(new_approxdp, "approxDP_func")

        if not fdp_off:  # Direct implementation of fDP
            # Tudor: i'll fix these
            fun1 = lambda x: fdp_bank.log_one_minus_fdp_gaussian(  # noqa: E731
                {"sigma": sigma}, x
            )
            fun2 = lambda x: fdp_bank.log_neg_fdp_grad_gaussian(  # noqa: E731
                {"sigma": sigma}, x
            )
            self.propagate_updates([fun1, fun2], "fDP_and_grad_log")
            # overwrite the fdp computation with the direct computation
            self.fdp = lambda x: fdp_bank.fDP_gaussian(
                {"sigma": sigma}, x
            )  # noqa: E731

        # the fDP of gaussian mechanism is equivalent to analytical calibration of approxdp,
        # so it should have been automatically handled numerically above

        # Discussion:  Sometimes delta as a function of eps has a closed-form solution
        # while eps as a function of delta does not
        # Shall we represent delta as a function of eps instead?
