# future
from __future__ import annotations

# stdlib
import secrets
from typing import Callable
from typing import List
from typing import TYPE_CHECKING
from typing import Tuple
from typing import Union

# third party
import jax
from jax import numpy as jnp
import numpy as np

# relative
from ..tensor.fixed_precision_tensor import FixedPrecisionTensor
from ..tensor.lazy_repeat_array import lazyrepeatarray
from ..tensor.passthrough import PassthroughTensor  # type: ignore
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
    state_tree: dict[int, "GammaTensor"],
    ledger: DataSubjectLedger,
    get_budget_for_user: Callable,
    deduct_epsilon_for_user: Callable,
    sigma: float,
    is_linear: bool = True,
    output_func: Callable = lambda x: x,
) -> Union[np.ndarray, jax.numpy.DeviceArray]:
    # relative
    from ..tensor.autodp.gamma_tensor import GammaTensor

    input_tensors = list(state_tree.values())
    # TODO: Vectorize this to return a larger GammaTensor instead of a list of Tensors
    for tensor in input_tensors:
        if tensor.func_str != "no_op":
            input_tensors +=

        else:
            # this is a private tensor
            # TODO: Double check with Andrew if this is correct- if we use the individual min/max values

            # Calculate everything needed for RDP
            if isinstance(input_tensor.child, FixedPrecisionTensor):
                value = input_tensor.child.decode()
            else:
                value = input_tensor.child

            while isinstance(value, PassthroughTensor):
                value = value.child

            if isinstance(input_tensor.min_vals, lazyrepeatarray):
                min_val_array = input_tensor.min_vals.to_numpy()
            else:
                min_val_array = input_tensor.min_vals

            if isinstance(input_tensor.max_vals, lazyrepeatarray):
                max_val_array = input_tensor.max_vals.to_numpy()
            else:
                max_val_array = input_tensor.max_vals

            l2_norms, l2_norm_bounds, sigmas, coeffs = calculate_bounds_for_mechanism(
                value_array=value,
                min_val_array=min_val_array,
                max_val_array=max_val_array,
                sigma=sigma,
            )

            if is_linear:
                lipschitz_bounds = coeffs.copy()
            else:
                lipschitz_bounds = input_tensor.lipschitz_bound

            if isinstance(input_tensor.data_subjects, np.ndarray):
                # Need to convert newDSL to old DSL-> this works because for each step we only use 1 data subject at a time
                while isinstance(input_tensor, PassthroughTensor):
                    root_child = input_tensor.child
                    input_tensor = root_child
                input_entities = input_tensor.data_subjects
            elif isinstance(input_tensor.data_subjects, DataSubjectList):
                input_entities = input_tensor.data_subjects.data_subjects_indexed
            else:
                raise NotImplementedError(
                    f"Undefined behaviour for data subjects type: {type(input_tensor.data_subjects)}"
                )

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
                )
                # We had to flatten the mask so the code generalized for N-dim arrays, here we reshape it back
                reshaped_mask = mask.reshape(value.shape)

                if mask is None:
                    raise Exception("Failed to publish; mask is None")

                # multiply values by the inverted mask
                filtered_input_tensor = value * (
                    1 - reshaped_mask
                )




                filtered_inputs.append(filtered_input_tensor)
                # output = np.asarray(output_func(filtered_inputs) + noise)

            except Exception as e:
                # stdlib
                import traceback

                print(traceback.format_exc())
                print(f"Failed to run vectorized_publish. {e}")

    print("We have filtered all the input tensors. Now to compute the result:")

    original_output = np.asarray(output_func(filtered_inputs))
    print("original output (before noise:", original_output)
    noise = np.asarray(
        [secrets.SystemRandom().gauss(0, sigma) for _ in range(original_output.size)]
    )
    noise.resize(original_output.shape)
    print("noise: ", noise)

    output = np.asarray(output_func(filtered_inputs) + noise)

    print("got output", output, type(output), output.dtype)

    return output.squeeze()
