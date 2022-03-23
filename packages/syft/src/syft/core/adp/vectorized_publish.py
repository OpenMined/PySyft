# stdlib
from random import gauss
from typing import Any
from typing import Callable
from typing import Tuple

# third party
import numpy as np

# relative
from .data_subject_ledger import DataSubjectLedger
from .data_subject_ledger import RDPParams
from .entity_list import EntityList


def calculate_bounds_for_mechanism(
    value_array: np.ndarray, min_val_array: np.ndarray, max_val_array: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculates the squared L2 norm values needed to create a Mechanism, and calculate
    privacy budget + spend. If you calculate the privacy budget spend with the worst
    case bound, you can show this number to the DS. If you calculate it with the
    regular value (the value computed below when public_only = False, you cannot show
    the privacy budget to the DS because this violates privacy."""

    # TODO: Double check whether the iDPGaussianMechanism class squares its
    # squared_l2_norm values!!

    # min_val_array = min_val_array.astype(np.int64)
    # max_val_array = max_val_array.astype(np.int64)

    # using np.ones_like dtype=value_array.dtype because without it the output was
    # of type "O" python object causing issues when doing operations against JAX
    worst_case_l2_norm = np.sqrt(
        np.sum(np.square(max_val_array - min_val_array))
    ) * np.ones_like(
        value_array
    )  # dtype=value_array.dtype)

    l2_norm = np.sqrt(np.sum(np.square(value_array))) * np.ones_like(value_array)
    # dtype=value_array.dtype
    #
    # print(l2_norm.shape, worst_case_l2_norm.shape)
    # print(l2_norm.shape)
    return l2_norm, worst_case_l2_norm


def vectorized_publish(
    node: Any,
    min_vals: np.ndarray,
    max_vals: np.ndarray,
    values: np.ndarray,
    data_subjects: EntityList,
    ledger: DataSubjectLedger,
    is_linear: bool = True,
    sigma: float = 1.5,
    output_func: Callable = np.sum
    # private: bool = False
) -> np.ndarray:
    # TODO convert values to np.int64
    # print("values1", type(values), values.dtype)
    # values = jnp.array(values, dtype=np.int64)
    # print("values2", type(values), values.dtype)

    print(f"Starting vectorized publish: {type(ledger)}")
    # Get all unique entities
    unique_data_subjects = data_subjects.one_hot_lookup
    # unique_data_subject_indices = np.arange(
    _ = np.arange(
        len(unique_data_subjects)
    )  # because unique_data_subjects returns an array, but we need indices

    print("Obtained data subject indices")

    # Calculate everything needed for RDP
    sigmas = np.reshape(np.ones_like(values) * sigma, -1)
    coeffs = np.ones_like(values).reshape(-1)
    l2_norms, l2_norm_bounds = calculate_bounds_for_mechanism(
        value_array=values, min_val_array=min_vals, max_val_array=max_vals
    )

    if is_linear:
        lipschitz_bounds = np.ones_like(values).reshape(-1)
    else:
        raise Exception("gamma_tensor.lipschitz_bound property would be used here")

    input_entities = data_subjects.entities_indexed[0].reshape(-1)

    print("Obtained all parameters for RDP")

    print("Initialized ledger!")
    ledger.entity_ids = np.array(input_entities, dtype=np.int64)

    # Query budget spend of all unique entities
    rdp_params = RDPParams(
        sigmas=sigmas,
        l2_norms=l2_norms,
        l2_norm_bounds=l2_norm_bounds,
        Ls=lipschitz_bounds,
        coeffs=coeffs,
    )

    try:
        mask = ledger.get_entity_overbudget_mask_for_epsilon_and_append(
            unique_entity_ids_query=input_entities,
            rdp_params=rdp_params,
            private=True,
            node=node,
        )

        if mask is None:
            raise Exception("Failed to publish mask is None")

        print("Obtained overbudgeted entity mask", mask.dtype)

        # Filter results
        filtered_inputs = values * (
            mask ^ 1
        )  # + gauss(0, sigma)  # Double check that noise has mean of 0
        output = np.asarray(output_func(filtered_inputs) + gauss(0, sigma))
        print("got output", type(output), output.dtype)
        return output
    except Exception as e:
        # stdlib
        import traceback

        print(traceback.format_exc())
        print(f"Failed to run vectorized_publish. {e}")
