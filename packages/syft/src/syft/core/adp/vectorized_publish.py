# stdlib
import secrets
from time import time
from typing import Callable
from typing import Tuple

# third party
import jax
from jax import numpy as jnp
import numpy as np

# relative
from .data_subject_ledger import DataSubjectLedger
from .data_subject_ledger import RDPParams
from .data_subject_list import DataSubjectList

# def calculate_bounds_for_mechanism(
#     value_array: np.ndarray, min_val_array: np.ndarray, max_val_array: np.ndarray
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """Calculates the squared L2 norm values needed to create a Mechanism, and calculate
#     privacy budget + spend. If you calculate the privacy budget spend with the worst
#     case bound, you can show this number to the DS. If you calculate it with the
#     regular value (the value computed below when public_only = False, you cannot show
#     the privacy budget to the DS because this violates privacy."""

#     # TODO: Double check whether the iDPGaussianMechanism class squares its
#     # squared_l2_norm values!!

#     # min_val_array = min_val_array.astype(np.int64)
#     # max_val_array = max_val_array.astype(np.int64)

#     # using np.ones_like dtype=value_array.dtype because without it the output was
#     # of type "O" python object causing issues when doing operations against JAX
#     worst_case_l2_norm = np.sqrt(
#         np.sum(np.square(max_val_array - min_val_array))
#     ) * np.ones_like(
#         value_array
#     )  # dtype=value_array.dtype)

#     l2_norm = np.sqrt(np.sum(np.square(value_array))) * np.ones_like(value_array)
#     # dtype=value_array.dtype
#     #
#     # print(l2_norm.shape, worst_case_l2_norm.shape)
#     # print(l2_norm.shape)
#     return l2_norm, worst_case_l2_norm


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
    values: np.ndarray,
    data_subjects: DataSubjectList,
    ledger: DataSubjectLedger,
    get_budget_for_user: Callable,
    deduct_epsilon_for_user: Callable,
    is_linear: bool = True,
    sigma: float = 1.5,
    output_func: Callable = np.sum,
    # private: bool = False
) -> np.ndarray:
    # TODO convert values to np.int64
    if values.dtype != np.int64:
        raise Exception("Values is not np.int64", values.dtype)

    print(f"Starting vectorized publish: {type(ledger)}")
    # Get all unique entities
    # unique_data_subjects = data_subjects.one_hot_lookup
    # unique_data_subject_indices = np.arange(

    print("Starting RDP Params Calculation")

    t1 = time()
    # Calculate everything needed for RDP

    l2_norms, l2_norm_bounds, sigmas, coeffs = calculate_bounds_for_mechanism(
        value_array=values, min_val_array=min_vals, max_val_array=max_vals, sigma=sigma
    )

    if is_linear:
        lipschitz_bounds = coeffs.copy()
    else:
        raise Exception("gamma_tensor.lipschitz_bound property would be used here")

    input_entities = data_subjects.data_subjects_indexed[0].reshape(-1)
    t2 = time()
    print("Obtained RDP Params, calculation time", t2 - t1)

    # Query budget spend of all unique entities
    rdp_params = RDPParams(
        sigmas=sigmas,
        l2_norms=l2_norms,
        l2_norm_bounds=l2_norm_bounds,
        Ls=lipschitz_bounds,
        coeffs=coeffs,
    )
    print("Finished RDP Params Initialization")
    try:
        # query and save
        mask = ledger.get_entity_overbudget_mask_for_epsilon_and_append(
            unique_entity_ids_query=input_entities,
            rdp_params=rdp_params,
            private=True,
            get_budget_for_user=get_budget_for_user,
            deduct_epsilon_for_user=deduct_epsilon_for_user,
        )
        # We had to flatten the mask so the code generalized for N-dim arrays, here we reshape it back
        reshaped_mask = mask.reshape(values.shape)
        print("Fixed mask shape!")
        # here we have the final mask and highest possible spend has been applied
        # to the data scientists budget field in the database

        if mask is None:
            raise Exception("Failed to publish mask is None")

        print("Obtained overbudgeted entity mask", mask.dtype)

        # multiply values by the inverted mask
        filtered_inputs = values * (
            1 - reshaped_mask
        )  # + gauss(0, sigma)  # Double check that noise has mean of 0
        noise = secrets.SystemRandom().gauss(0, sigma)
        output = np.asarray(output_func(filtered_inputs) + noise)
        print("got output", type(output), output.dtype)
        return output
    except Exception as e:
        # stdlib
        import traceback

        print(traceback.format_exc())
        print(f"Failed to run vectorized_publish. {e}")
