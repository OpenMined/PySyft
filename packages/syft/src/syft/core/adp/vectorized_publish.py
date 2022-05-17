# future
from __future__ import annotations

# stdlib
import secrets
from typing import Callable
from typing import List
from typing import Optional
from typing import TYPE_CHECKING
from typing import Tuple
from typing import Union

# third party
import jax
from jax import numpy as jnp
import numpy as np
from tqdm import tqdm

# relative
from .data_subject_ledger import DataSubjectLedger
from .data_subject_ledger import RDPParams
from .data_subject_list import DataSubjectList

if TYPE_CHECKING:
    # relative
    from ..tensor.autodp.gamma_tensor import GammaTensor


@jax.jit
def calculate_bounds_for_mechanism(
    value_array: jnp.ndarray,
    min_val_array: jnp.ndarray,
    max_val_array: jnp.ndarray,
    sigma: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.array, jnp.array]:
    ones_like = jnp.ones_like(value_array)
    one_dim = jnp.reshape(ones_like, -1)

    worst_case_l2_norm = (
        jnp.sqrt(jnp.sum(jnp.square(max_val_array - min_val_array))) * one_dim
    )

    l2_norm = jnp.sqrt(jnp.sum(jnp.square(value_array))) * one_dim
    return l2_norm, worst_case_l2_norm, one_dim * sigma, one_dim


def vectorized_publish(
    min_vals: np.ndarray,
    max_vals: np.ndarray,
    state_tree: dict[int, "GammaTensor"],
    data_subjects: DataSubjectList,
    ledger: DataSubjectLedger,
    get_budget_for_user: Callable,
    deduct_epsilon_for_user: Callable,
    is_linear: bool = True,
    sigma: float = 1.5,
    output_func: Callable = lambda x: x,
    fpt_encode_func: Optional[Callable] = None,
) -> Union[np.ndarray, jax.numpy.DeviceArray]:
    """
    Steps:
    1. Get all the input tensors with private data that helped create the tensor we're publishing.
    2. For every tensor with raw private data, we need to:
        - Collect the parameters needed to calculate the privacy budget spend using RDP
        - Calculate the privacy budget spend for each data subject in that tensor
        - Filter out data for any individual data subject that doesn't have enough PB to be involved in the query
    3. Recalculate the query now that we know we're not including data from anyone over their PB
    4. Add noise to the result of the query
    """


    # relative
    from ..tensor.autodp.gamma_tensor import GammaTensor

    # TODO: Vectorize this to return a larger GammaTensor instead of a list of Tensors
    input_tensors: List[GammaTensor] = GammaTensor.get_input_tensors(state_tree)

    filtered_inputs = []

    # This will reveal the # of input tensors to the user- remove this before merging to dev
    for input_tensor in tqdm(input_tensors):
        # TODO: Double check with Andrew if this is correct- if we use the individual min/max values

        # Parameters needed for PB calculation as seen here: https://arxiv.org/abs/2008.11193
        l2_norms, l2_norm_bounds, sigmas, coeffs = calculate_bounds_for_mechanism(
            value_array=input_tensor.value,
            min_val_array=input_tensor.min_val,
            max_val_array=input_tensor.max_val,
            sigma=sigma,
        )

        if is_linear:  # 2.7 from https://arxiv.org/abs/2008.11193
            lipschitz_bounds = coeffs.copy()
        else:  # 2.8 from https://arxiv.org/abs/2008.11193
            lipschitz_bounds = input_tensor.lipschitz_bound

        input_data_subjects = input_tensor.data_subjects.data_subjects_indexed

        rdp_params = RDPParams(
            sigmas=sigmas,
            l2_norms=l2_norms,
            l2_norm_bounds=l2_norm_bounds,
            Ls=lipschitz_bounds,
            coeffs=coeffs,
        )

        try:
            # This method spends the privacy budget needed by this query
            mask = ledger.get_entity_overbudget_mask_for_epsilon_and_append(
                unique_entity_ids_query=input_data_subjects,
                rdp_params=rdp_params,
                private=True,
                get_budget_for_user=get_budget_for_user,
                deduct_epsilon_for_user=deduct_epsilon_for_user,
            )
            # We had to flatten the mask so the code generalized for N-dim arrays, here we reshape it back
            reshaped_mask = mask.reshape(input_tensor.value.shape)

            if mask is None:
                raise Exception("Failed to publish mask")

            # multiply values by the inverted mask; the only data you see is from data subjects that are UNDER budget
            filtered_input_tensor = input_tensor.value * (
                1 - reshaped_mask
            )

            filtered_inputs.append(filtered_input_tensor)

        except Exception as e:
            # stdlib
            import traceback

            print(traceback.format_exc())
            print(f"Failed to run vectorized_publish. {e}")

    print("We have filtered all the input tensors. Now to compute the result:")

    print("Filtered inputs ", filtered_inputs)
    original_output = np.asarray(output_func(filtered_inputs))
    print("original output (before noise:", original_output)
    noise = np.asarray(
        [secrets.SystemRandom().gauss(0, sigma) for _ in range(original_output.size)]
    )
    noise.resize(original_output.shape)
    print("noise: ", noise)
    if fpt_encode_func is not None:
        noise = fpt_encode_func(noise)
        print("Noise after FPT", noise)
    output = np.asarray(output_func(filtered_inputs) + noise)
    print("got output", type(output), output.dtype)

    # TODO: Need to decode the FPT results
    return output.squeeze()
