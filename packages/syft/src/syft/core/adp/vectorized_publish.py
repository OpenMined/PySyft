# future
from __future__ import annotations

# stdlib
import secrets
from signal import Sigmasks
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
from sympy import Float
from tqdm import tqdm

# relative
from .data_subject_ledger import DataSubjectLedger
from .data_subject_ledger import RDPParams
from .data_subject_ledger import load_cache
from .data_subject_ledger import compute_rdp_constant
from .data_subject_ledger import convert_constants_to_indices


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
    
    if "float" not in str(sigma):
        raise RuntimeError(f"Sigma is of type {type(sigma)}. Expected type float.")

    return l2_norm, worst_case_l2_norm, one_dim * sigma, one_dim

def calibrate_sigma(
    cache_constant2epsilon: np.ndarray,
    value: np.ndarray,
    min_vals: np.ndarray,
    max_vals: np.ndarray,
    is_linear: bool,
    query_limit: int=5,
    sigma=1.5,
   ) -> np.ndarray:

    """ 
       Adjust the value of sigma chosen to have a 90% chance of being less than query_limit
 
    """
    
    # RDP Params Initialization
    l2_norms, l2_norm_bounds, sigmas, coeffs = calculate_bounds_for_mechanism(
        value_array=value,
        min_val_array=min_vals,
        max_val_array=max_vals,
        sigma=sigma,
    )
    print("Query Limit: ",query_limit )
    print("Sigma before Calibration:\n ", sigma)

    if is_linear:
        lipschitz_bounds = coeffs.copy()
    else:
        lipschitz_bounds = value.lipschitz_bound

    # Query budget spend of all data_subjects
    rdp_params = RDPParams(
        sigmas=sigmas,
        l2_norms=l2_norms,
        l2_norm_bounds=l2_norm_bounds,
        Ls=lipschitz_bounds,
        coeffs=coeffs,
    )
    rdp_constants = compute_rdp_constant(rdp_params, private=False)

    rdp_constants_lookup = convert_constants_to_indices(rdp_constants)
    max_rdp = np.max(rdp_constants_lookup)
   
    budget_spend = cache_constant2epsilon.take((max_rdp).astype(np.int64))
    
    if budget_spend >= query_limit:
        print("****** Budget spent EXCEEDS limit set by data owner")
        # This is the first index in the cache that has an epsilon >= query limit
        threshold_index = np.searchsorted(cache_constant2epsilon, query_limit)
        # There is a 90% chance the final budget spent will be below the query_limit
        selected_rdp_value = np.random.choice(np.arange(threshold_index - 9, threshold_index + 1))
        calibrated_sigma = selected_rdp_value * np.sqrt(max_rdp / selected_rdp_value)

    else:
        print("****** Budget spent BELOW limit set by data owner")
        calibrated_sigma = rdp_params.sigmas[0]

    print("Sigma after Calibration: \n", calibrated_sigma)

    return calibrated_sigma


def vectorized_publish(
        value: np.ndarray,
    min_vals: np.ndarray,
    max_vals: np.ndarray,
    state_tree: dict[int, "GammaTensor"],
    ledger: DataSubjectLedger,
    get_budget_for_user: Callable,
    deduct_epsilon_for_user: Callable,
    is_linear: bool = True,
    sigma: float = 1.5,
    output_func: Callable = lambda x: x,
    fpt_encode_func: Optional[Callable] = None,
    query_limit: int = 999999
) -> Union[np.ndarray, jax.numpy.DeviceArray]:
    # relative
    from ..tensor.autodp.gamma_tensor import GammaTensor

    # # TODO: Use calibration here
    CONSTANT2EPSILSON_CACHE_FILENAME = "constant2epsilon_1200k.npy"
    _cache_constant2epsilon = load_cache(filename=CONSTANT2EPSILSON_CACHE_FILENAME)

    calibrated_sigma = calibrate_sigma(
        cache_constant2epsilon=_cache_constant2epsilon,
        value=value,
        min_vals=min_vals,
        max_vals=max_vals,
        is_linear=is_linear,
        query_limit=query_limit,
        sigma=sigma)

    input_tensors: List[GammaTensor] = GammaTensor.get_input_tensors(state_tree)

    filtered_inputs = []

    # This will reveal the # of input tensors to the user- remove this before merging to dev
    for input_tensor in tqdm(input_tensors):
        # TODO: Double check with Andrew if this is correct- if we use the individual min/max values

        #Calculate everything needed for RDP
        l2_norms, l2_norm_bounds, sigmas, coeffs = calculate_bounds_for_mechanism(
            value_array=input_tensor.value,
            min_val_array=input_tensor.min_val,
            max_val_array=input_tensor.max_val,
            sigma=calibrated_sigma,
        )

        if is_linear:
            lipschitz_bounds = coeffs.copy()
        else:
            lipschitz_bounds = input_tensor.lipschitz_bound
            # raise Exception("gamma_tensor.lipschitz_bound property would be used here")

        input_entities = input_tensor.data_subjects.data_subjects_indexed
        # data_subjects.data_subjects_indexed[0].reshape(-1)
        # t2 = time()

        # Query budget spend of all unique data_subjects
        rdp_params = RDPParams(
            sigmas=sigmas,
            l2_norms=l2_norms,
            l2_norm_bounds=l2_norm_bounds,
            Ls=lipschitz_bounds,
            coeffs=coeffs,
        )
        # print("Finished RDP Params Initialization")
        try:
            # query and save
            mask = ledger.get_entity_overbudget_mask_for_epsilon_and_append(
                unique_entity_ids_query=input_entities,
                rdp_params=rdp_params,
                private=True,
                get_budget_for_user=get_budget_for_user,
                deduct_epsilon_for_user=deduct_epsilon_for_user,
                cache_constant2epsilon = _cache_constant2epsilon
            )
            # We had to flatten the mask so the code generalized for N-dim arrays, here we reshape it back
            reshaped_mask = mask.reshape(input_tensor.value.shape)
            # print("Fixed mask shape!")
            # here we have the final mask and highest possible spend has been applied
            # to the data scientists budget field in the database

            if mask is None:
                raise Exception("Failed to publish mask is None")

            # print("Obtained overbudgeted entity mask", mask.dtype)

            # multiply values by the inverted mask
            filtered_input_tensor = input_tensor.value * (
                1 - reshaped_mask
            )  # + gauss(0, sigma)  # Double check that noise has mean of 0

            filtered_inputs.append(filtered_input_tensor)
            # output = np.asarray(output_func(filtered_inputs) + noise)

        except Exception as e:
            # stdlib
            import traceback

            print(traceback.format_exc())
            print(f"Failed to run vectorized_publish. {e}")

    #print("We have filtered all the input tensors. Now to compute the result:")

    # noise = secrets.SystemRandom().gauss(0, sigma)
    #print("Filtered inputs ", filtered_inputs)
    original_output = np.asarray(output_func(filtered_inputs))
    #print("original output (before noise:", original_output)
    noise = np.asarray(
        [secrets.SystemRandom().gauss(0, sigma) for _ in range(original_output.size)]
    )
    noise.resize(original_output.shape)
    #print("noise: ", noise)
    if fpt_encode_func is not None:
        noise = fpt_encode_func(noise)
        #print("Noise after FPT", noise)
    output = np.asarray(output_func(filtered_inputs) + noise)
    #print("got output", type(output), output.dtype)
    return output.squeeze()
