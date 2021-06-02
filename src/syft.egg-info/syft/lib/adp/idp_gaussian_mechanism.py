# stdlib
from typing import Dict

# third party
from autodp import dp_bank
from autodp import fdp_bank
from autodp.autodp_core import Mechanism
import numpy as np


def individual_RDP_gaussian(params: Dict, alpha: float) -> np.float64:
    """
    :param params:
        'sigma' --- is the normalized noise level: std divided by global L2 sensitivity
    :param alpha: The order of the Renyi Divergence
    :return: Evaluation of the RDP's epsilon
    """
    sigma = params["sigma"]
    value = params["value"]
    L = params["L"]
    assert sigma > 0
    assert alpha >= 0

    return (alpha * (L ** 2) * (value ** 2)) / (2 * (sigma ** 2))


# Example of a specific mechanism that inherits the Mechanism class
class iDPGaussianMechanism(Mechanism):
    def __init__(
        self,
        sigma: float,
        value: float,
        L: np.float,
        entity: str,
        name: str = "Gaussian",
        RDP_off: bool = False,
        approxDP_off: bool = False,
        fdp_off: bool = True,
        use_basic_rdp_to_approx_dp_conversion: bool = False,
        use_fdp_based_rdp_to_approx_dp_conversion: bool = False,
    ):
        # the sigma parameter is the std of the noise divide by the l2 sensitivity
        Mechanism.__init__(self)

        self.name = name  # When composing
        self.params = {
            "sigma": sigma,
            "value": value,
            "L": L,
        }  # This will be useful for the Calibrator
        self.entity = entity
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
