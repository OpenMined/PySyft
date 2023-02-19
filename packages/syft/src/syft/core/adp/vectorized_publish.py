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
    for data_subject, rdp_constant in rdp_constants.items():
        # convert back to numpy for serde
        data_subject_rdp_constants[data_subject] = np.array(
            max(
                data_subject_rdp_constants.get(data_subject, -np.inf),
                rdp_constant,
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
    from ..tensor.autodp.new_phi_tensor import RowPhiTensors
    from ..tensor.autodp.phi_tensor import PhiTensor
    
    # create a copy so we can recompute the final value after filtering
    tensor_copy = tensor

    # traverse the computation tree to get the phi_tensors
    phi_tensors = tensor_copy.sources

    # For each PhiTensor we either need the lipschitz bound or we can use
    # one to keep the same formula for the linear queries
    lipschitz_bound = 1 if is_linear else tensor.lipschitz_bound

    # compute the rdp constant for each phi tensor
    ds_2_data = {}
    ds_2_min_vals = {}
    ds_2_max_vals = {}
    ds_2_tensors = {}
    
    for phi_tensor_id in phi_tensors:
        phi_tensor = phi_tensors[phi_tensor_id]
        
        if isinstance(phi_tensor, PhiTensor):
            if str(phi_tensor.dtype) == "bool":
                phi_tensor = deepcopy(phi_tensor.astype("float"))
                
            data_subject = phi_tensor.data_subject
            
            child = phi_tensor.child
            if isinstance(child, FixedPrecisionTensor):
                child = child.decode()
            old_data = ds_2_data.get(data_subject, [])
            old_data.append(child)
            ds_2_data[data_subject] = old_data
            
            
            old_min_vals = ds_2_min_vals.get(data_subject, [])
            old_min_vals.append(phi_tensor.min_vals.to_numpy())
            ds_2_min_vals[data_subject] = old_min_vals
            
            old_max_vals = ds_2_max_vals.get(data_subject, [])
            old_max_vals.append(phi_tensor.max_vals.to_numpy())
            ds_2_max_vals[data_subject] = old_max_vals
            
            old_tensors = ds_2_tensors.get(data_subject, [])
            old_tensors.append(phi_tensor_id)
            ds_2_tensors[data_subject] = old_tensors
    
        elif isinstance(phi_tensor, RowPhiTensors):
            # TODO: add bool conversion and FixedPrecision Tensor unwrapping
            for i, data_subject in enumerate(phi_tensor.data_subjects):
                data = phi_tensor.data[i]
                min_vals, max_vals = phi_tensor.bounds[i].to_numpy()
                
                old_data = ds_2_data.get(data_subject, [])
                old_data.append(data)
                ds_2_data[data_subject] = old_data
                
                old_min_vals = ds_2_min_vals.get(data_subject, [])
                old_min_vals.append(min_vals)
                ds_2_min_vals[data_subject] = old_min_vals
                
                old_max_vals = ds_2_max_vals.get(data_subject, [])
                old_max_vals.append(max_vals)
                ds_2_max_vals[data_subject] = old_max_vals
                
                old_tensors = ds_2_tensors.get(data_subject, [])
                old_tensors.append(phi_tensor_id)
                ds_2_tensors[data_subject] = old_tensors
        else:
            raise Exception("ce plm")
    
    rdp_constants = {}
    for data_subject in ds_2_data:
        # TODO 0.8 optimize the computation of l2_norm_bounds
        child = np.array(ds_2_data[data_subject])
        min_vals = np.array(ds_2_min_vals[data_subject])
        max_vals = np.array(ds_2_max_vals[data_subject])

        l2_norms, l2_norm_bounds = calculate_bounds_for_mechanism(
            child,
            min_vals,
            max_vals,
        )
        param = RDPParams(
            sigmas=sigma,
            l2_norms=l2_norms,
            l2_norm_bounds=l2_norm_bounds,
            Ls=lipschitz_bound,
        )
        rdp_constants[data_subject] = compute_rdp_constant(
            rdp_params=param, private=private
        )

    # get epsilon for each tensor based on the rdp constant
    epsilons = {
        data_subject: ledger._get_epsilon_spend(
            np.array([rdp_constants[data_subject]])
        )  # this is kinda dumb
        for data_subject in ds_2_data.keys()
    }

    ds_kept = {data_subject: epsilons[data_subject] <= privacy_budget for data_subject in epsilons}
    epsilon_spend = max([epsilons[eps_id] * ds_kept[eps_id] for eps_id in epsilons])

    # In the previous iteration, PhiTensors would be filtered by being replaced with zeros.
    # If for RowPhiTensors we can remove them from the computations altogheter, then
    # we would need to recompute the Lipschitz bound, as that value might be modified 
    # if we remove the bounds from the filtered tensors in the optimization process.
    # We should probably considering filtering based on the neutral operands of each operation
    # so we could justify keeping the same value. Not that clear tho
   
    # compute the new output based on the filtered values
    if False in ds_kept.values():
        new_state = {}
        for phi_tensor_id in phi_tensors:
            phi_tensor = phi_tensors[phi_tensor_id]
            if isinstance(phi_tensor, PhiTensor):
                if ds_kept[phi_tensor.data_subject]:
                    new_state[phi_tensor_id] = phi_tensor.child
                else:
                    new_state[phi_tensor_id] = phi_tensor.filtered()
            
            elif isinstance(phi_tensor, RowPhiTensors):
                filtered_data_subjects = []
                for data_subject in phi_tensor.data_subjects:
                    if not ds_kept[data_subject]:
                        filtered_data_subjects.append(data_subject)
                        
                new_state[phi_tensor_id] = phi_tensor.filtered(filtered_data_subjects)
            else:
                raise Exception("ce plm")
                
        original_output = tensor.reconstruct(new_state)
    else:
        original_output = tensor.child
    return original_output, epsilon_spend, rdp_constants
