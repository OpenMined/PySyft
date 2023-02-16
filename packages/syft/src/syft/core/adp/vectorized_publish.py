# future
from __future__ import annotations

# stdlib
from copy import deepcopy
import secrets
from typing import Callable
from typing import Dict

# from typing import List
from typing import TYPE_CHECKING
from typing import Tuple

# third party
import jax
from jax import numpy as jnp
import numpy as np

# relative
from ...core.node.common.node_manager.user_manager import RefreshBudgetException
from ...core.tensor.fixed_precision_tensor import FixedPrecisionTensor
from ..tensor.passthrough import PassthroughTensor  # type: ignore
from .data_subject_ledger import DataSubjectLedger
from .data_subject_ledger import RDPParams
from .data_subject_ledger import compute_rdp_constant

if TYPE_CHECKING:
    # relative
    from ..tensor.autodp.gamma_tensor import GammaTensor


@jax.jit
def calculate_bounds_for_mechanism(
    value_array: jnp.ndarray,
    min_val_array: jnp.ndarray,
    max_val_array: jnp.ndarray,
) -> Tuple[jnp.array, jnp.array]:
    worst_case_l2_norm = jnp.sqrt(jnp.sum(jnp.square(max_val_array - min_val_array)))

    l2_norm = jnp.sqrt(jnp.sum(jnp.square(value_array)))
    return l2_norm, worst_case_l2_norm


def publish(
    tensor: GammaTensor,
    ledger: DataSubjectLedger,
    get_budget_for_user: Callable,
    deduct_epsilon_for_user: Callable,
    sigma: float,
    is_linear: bool = True,
    private: bool = True,
) -> np.ndarray:
    # root_child = None
    while isinstance(tensor, PassthroughTensor):
        root_child = tensor.child
        tensor = root_child

    privacy_budget = get_budget_for_user(verify_key=ledger.user_key)

    original_output, epsilon_spend, rdp_constants = compute_epsilon(
        tensor, privacy_budget, is_linear, sigma, private, ledger
    )

    raw_rdp_constants = np.array(list(rdp_constants.values()))

    if np.any(raw_rdp_constants < 0):
        raise Exception(
            "Negative budget spend not allowed in PySyft for safety reasons."
            "Please contact the OpenMined support team for help."
            "For that you can either:"
            " * describe your issue on our Slack #support channel. To join: https://openmined.slack.com/"
            " * send us an email describing your problem at support@openmined.org"
            " * leave us an issue here: https://github.com/OpenMined/PySyft/issues"
        )

    if jnp.isnan(epsilon_spend):
        raise Exception("Epsilon is NaN")

    if jnp.any(jnp.isnan(raw_rdp_constants)):
        raise Exception("RDP constant in NaN")

    if jnp.any(jnp.isinf(raw_rdp_constants)):
        raise Exception("RDP constant in inf")

    if epsilon_spend < 0:
        raise Exception(
            "Negative budget spend not allowed in PySyft for safety reasons."
            "Please contact the OpenMined support team for help."
            "\nFor that you can either:"
            "\n * describe your issue on our Slack #support channel. To join: https://openmined.slack.com/"
            "\n * send us an email describing your problem at support@openmined.org"
            "\n * leave us an issue here: https://github.com/OpenMined/PySyft/issues"
        )

    # The user spends their privacy budget before getting the result
    attempts = 0
    while attempts < 5:
        attempts += 1
        try:
            ledger.spend_epsilon(
                deduct_epsilon_for_user=deduct_epsilon_for_user,
                epsilon_spend=epsilon_spend,
                old_user_budget=privacy_budget,
            )
            break
        except RefreshBudgetException:  # nosec
            ledger.spend_epsilon(
                deduct_epsilon_for_user=deduct_epsilon_for_user,
                epsilon_spend=epsilon_spend,
                old_user_budget=privacy_budget,
            )

        except Exception as e:
            print(f"Problem spending epsilon. {e}")
            raise e

    # The RDP constants are adjusted to account for the amount of exposure every
    # data subject's data has had.
    data_subject_rdp_constants: Dict[str, np.ndarray] = {}
    for tensor_id in rdp_constants:
        data_subject = tensor.sources[tensor_id].data_subject.to_string()
        # convert back to numpy for serde
        data_subject_rdp_constants[data_subject] = np.array(
            max(
                data_subject_rdp_constants.get(data_subject, -np.inf),
                rdp_constants[tensor_id],
            )
        )

    ledger.update_rdp_constants(data_subject_rdp_constants=data_subject_rdp_constants)
    ledger._write_ledger()

    # We sample noise from a cryptographically secure distribution
    # TODO(0.8): Replace with discrete gaussian distribution instead of regular
    # gaussian to eliminate floating pt vulns
    noise = np.asarray(
        [secrets.SystemRandom().gauss(0, sigma) for _ in range(original_output.size)]
    ).reshape(original_output.shape)

    return original_output + noise


# TODO(0.8): this function should be vectorized for jax
def compute_epsilon(
    tensor: GammaTensor,
    privacy_budget: float,
    is_linear: bool,
    sigma: float,
    private: bool,
    ledger: DataSubjectLedger,
) -> Tuple:
    # create a copy so we can recompute the final value after filtering
    tensor_copy = deepcopy(tensor)

    # traverse the computation tree to get the phi_tensors
    phi_tensors = tensor_copy.sources

    # For each PhiTensor we either need the lipschitz bound or we can use
    # one to keep the same formula for the linear queries
    lipschitz_bound = 1 if is_linear else tensor.lipschitz_bound

    # compute the rdp constant for each phi tensor
    rdp_constants = {}
    for phi_tensor_id in phi_tensors:
        # TODO 0.8: figure a way to iterate over data_subjects and group phi_tensors when computing
        # the Lipschitz bound
        # TODO 0.8 optimize the computation of l2_norm_bounds
        phi_tensor = phi_tensors[phi_tensor_id]

        # handle bools
        if str(phi_tensor.dtype) == "bool":
            phi_tensor = deepcopy(phi_tensor.astype("float"))

        child = phi_tensor.child
        min_vals = phi_tensor.min_vals.to_numpy()
        max_vals = phi_tensor.max_vals.to_numpy()

        # handle FixedPrecisionTensor
        if isinstance(child, FixedPrecisionTensor):
            child = child.decode()

        l2_norms, l2_norm_bounds = calculate_bounds_for_mechanism(
            child,
            min_vals,
            max_vals,
        )  # , tensor.min_vals.to_numpy(), tensor.max_vals.to_numpy())
        param = RDPParams(
            sigmas=sigma,
            l2_norms=l2_norms,
            l2_norm_bounds=l2_norm_bounds,
            Ls=lipschitz_bound,
        )
        rdp_constants[phi_tensor_id] = compute_rdp_constant(
            rdp_params=param, private=private
        )

    # get epsilon for each tensor based on the rdp constant
    epsilons = {
        phi_tensor_id: ledger._get_epsilon_spend(
            np.array([rdp_constants[phi_tensor_id]])
        )  # this is kinda dumb
        for phi_tensor_id in phi_tensors
    }

    filtered = {eps: epsilons[eps] <= privacy_budget for eps in epsilons}
    epsilon_spend = max([epsilons[eps_id] * filtered[eps_id] for eps_id in epsilons])

    new_state = {
        phi_tensor_id: phi_tensor.child
        if filtered[phi_tensor_id]
        else jnp.zeros_like(phi_tensors[phi_tensor_id].child)
        for phi_tensor_id, phi_tensor in phi_tensors.items()
    }

    # compute the new output based on the filtered values
    if False in filtered.values():
        original_output = tensor.reconstruct(new_state)
    else:
        original_output = tensor.child
    return original_output, epsilon_spend, rdp_constants
