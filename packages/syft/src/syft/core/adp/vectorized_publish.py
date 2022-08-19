# future
from __future__ import annotations

# stdlib
import secrets
from typing import Callable
from typing import List
from typing import TYPE_CHECKING
from typing import Tuple
from typing import Union
from copy import deepcopy

# third party
import jax
from jax import numpy as jnp
import numpy as np

# relative
from ...core.node.common.node_manager.user_manager import RefreshBudgetException
from ..tensor.fixed_precision_tensor import FixedPrecisionTensor
from ..tensor.lazy_repeat_array import lazyrepeatarray
from ..tensor.passthrough import PassthroughTensor  # type: ignore
from .data_subject_ledger import DataSubjectLedger
from .data_subject_ledger import RDPParams
from .data_subject_ledger import compute_rdp_constant
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


def publish(
    tensor: GammaTensor,
    ledger: DataSubjectLedger,
    get_budget_for_user: Callable,
    deduct_epsilon_for_user: Callable,
    sigma: float,
    is_linear: bool = True,
) -> np.ndarray:
    """
    This method applies Individual Differential Privacy (IDP) as defined in https://arxiv.org/abs/2008.11193
        - Key results: Theorem 2.7 and 2.8 show how much privacy budget is spent by a query.

    Given a tensor, it checks if the user (a data scientist) has enough privacy budget (PB) to see data from every data
    subject.
    - If the user has enough privacy budget, then DP noise is directly added to the result, and PB is deducted.
    - If the user doesn't have enough PB, then every data subject with a higher epsilon than their PB's data is removed.
        - The epsilons are then recomputed to see if the user now has enough PB to see the remaining data or not.


    Notes:
        - The privacy budget spent by a query equals the maximum epsilon increase of any data subject in the dataset.
    """

    # Step 0: Ensure our Tensor's private data is in a form that is usable.
    if isinstance(tensor.child, FixedPrecisionTensor):
        # Incase SMPC is involved, there will be an FPT in the chain to account for
        value = tensor.child.decode()
    else:
        value = tensor.child

    while isinstance(value, PassthroughTensor):
        value = value.child

    # Step 1: We obtain all the parameters needed to calculate Epsilons
    if isinstance(tensor.min_vals, lazyrepeatarray):
        min_val_array = tensor.min_vals.to_numpy()
    else:
        min_val_array = tensor.min_vals

    if isinstance(tensor.max_vals, lazyrepeatarray):
        max_val_array = tensor.max_vals.to_numpy()
    else:
        max_val_array = tensor.max_vals

    if isinstance(tensor.data_subjects, np.ndarray):
        root_child = None
        while isinstance(tensor, PassthroughTensor):
            root_child = tensor.child
            tensor = root_child
        input_entities = tensor.data_subjects
    elif isinstance(tensor.data_subjects, DataSubjectList):
        input_entities = tensor.data_subjects.data_subjects_indexed
    else:
        raise NotImplementedError(
            f"Undefined behaviour for data subjects type: {type(tensor.data_subjects)}"
        )

    l2_norms, l2_norm_bounds, sigmas, coeffs = calculate_bounds_for_mechanism(
        value_array=value,
        min_val_array=min_val_array,
        max_val_array=max_val_array,
        sigma=sigma,
    )

    if is_linear:
        lipschitz_bounds = coeffs.copy()
    else:
        lipschitz_bounds = tensor.lipschitz_bound

    rdp_params = RDPParams(
        sigmas=sigmas,
        l2_norms=l2_norms,
        l2_norm_bounds=l2_norm_bounds,
        Ls=lipschitz_bounds,
        coeffs=coeffs,
    )

    # Step 2: Calculate the epsilon spend for this query

    # rdp_constant = all terms in Theorem. 2.7 or 2.8 of https://arxiv.org/abs/2008.11193 EXCEPT alpha
    rdp_constants = compute_rdp_constant(rdp_params, private=True)
    all_epsilons = ledger.calculate_epsilon_spend(rdp_constants)  # This is the epsilon spend for ALL data subjects
    if any(all_epsilons < 0):
        raise Exception("Negative budget spend not allowed in PySyft for safety reasons. Please contact the OpenMined"
                        "support team for help.")

    epsilon_spend = max(all_epsilons)  # This is the epsilon spend for the QUERY, a single float.
    if epsilon_spend < 0:
        raise Exception("Negative budget spend not allowed in PySyft for safety reasons. Please contact the OpenMined"
                        "support team for help.")

    # Step 3: Check if the user has enough privacy budget for this query
    privacy_budget = get_budget_for_user(verify_key=ledger.user_key)
    has_budget = epsilon_spend <= privacy_budget

    # Step 4- Path 1: If the User has enough Privacy Budget, we just add noise, deduct budget, and return the result.
    if has_budget:
        original_output = tensor.child

        # We sample noise from a cryptographically secure distribution
        # TODO: Replace with discrete gaussian distribution instead of regular gaussian to eliminate floating pt vulns
        noise = np.asarray(
            [secrets.SystemRandom().gauss(0, sigma) for _ in range(original_output.size)]
        ).reshape(original_output.shape)

        # The user spends their privacy budget before getting the result
        attempts = 0
        while attempts < 5:
            attempts += 1
            try:
                ledger.spend_epsilon(
                    deduct_epsilon_for_user=deduct_epsilon_for_user,
                    epsilon_spend=epsilon_spend,
                    old_user_budget=privacy_budget
                )
                break
            except RefreshBudgetException:  # nosec
                ledger.spend_epsilon(
                    deduct_epsilon_for_user=deduct_epsilon_for_user,
                    epsilon_spend=epsilon_spend,
                    old_user_budget=privacy_budget
                )

            except Exception as e:
                print(f"Problem spending epsilon. {e}")
                raise e

        # The RDP constants are adjusted to account for the amount of exposure every data subject's data has had.
        ledger.update_rdp_constants(query_constants=rdp_constants, entity_ids_query=input_entities)
        ledger._write_ledger()
        return original_output + noise

    # Step 4- Path 2: User doesn't have enough privacy budget.
    elif not has_budget:
        # If the user doesn't have enough PB, they shouldn't see data of high epsilon data subjects (privacy violation)
        # So we will remove data belonging to these data subjects from the computation.

        # Step 4.1: Figure out which data subjects are within the PB & the highest possible spend
        within_budget_filter = jnp.ones_like(all_epsilons) * privacy_budget >= all_epsilons
        highest_possible_spend = jnp.max(all_epsilons * within_budget_filter)

        # Step 4.2: Figure out which Tensors in the Source dictionary have those data subjects
        filtered_sourcetree = deepcopy(tensor.state)
        input_tensors = list(filtered_sourcetree.values())
        parent_branch = [filtered_sourcetree for _ in input_tensors]  # TODO: Ensure this isn't deepcopying!
        for parent_state, input_tensor in zip(parent_branch, input_tensors):
            if input_tensor.func_str == "no_op":  # This is raw, unprocessed private data. Filter if eps spend > PB!
                # Calculate epsilon spend for this tensor
                l2_norms = jnp.sqrt(jnp.sum(jnp.square(input_tensor.child)))

                rdp_params = RDPParams(
                    sigmas=sigmas,
                    l2_norms=l2_norms,
                    l2_norm_bounds=l2_norm_bounds,
                    Ls=lipschitz_bounds,
                    coeffs=coeffs,
                )

                # Privacy loss associated with this private data specifically
                epsilon = max(ledger.calculate_epsilon_spend(compute_rdp_constant(rdp_params, private=True)))

                # Filter if > privacy budget
                if epsilon > privacy_budget:
                    filtered_tensor = input_tensor.filtered()

                    # Replace the original tensor with this filtered one.
                    parent_state[input_tensor.id] = filtered_tensor

                # If epsilon <= privacy budget, we don't need to do anything- the user has enough PB to use the data.
            else:
                input_tensors.append(list(input_tensor.state.values()))
                parent_branch += [input_tensor.state for _ in input_tensor.state.values()]

        # Recompute the tensor's value now that some of its inputs have been filtered, and repeat epsilon calculations
        new_tensor = tensor.swap_state(filtered_sourcetree)
        return new_tensor.publish(
            get_budget_for_user=get_budget_for_user,
            deduct_epsilon_for_user=deduct_epsilon_for_user,
            ledger=ledger,
            sigma=sigma
        )

    # Step 5: Revel in happiness.
    else:
        raise Exception


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
    from tqdm import tqdm

    input_tensors = List[GammaTensor] = GammaTensor.get_input_tensors(state_tree)
    # TODO: Vectorize this to return a larger GammaTensor instead of a list of Tensors

    filtered_inputs = []
    for input_tensor in tqdm(input_tensors):
        # TODO: Double check with Andrew if this is correct- if we use the individual min/max values

        # t1 = time()
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
            # raise Exception("gamma_tensor.lipschitz_bound property would be used here")

        if isinstance(input_tensor.data_subjects, np.ndarray):
            root_child = None
            while isinstance(input_tensor, PassthroughTensor):
                root_child = input_tensor.child
                input_tensor = root_child
            # input_entities = np.zeros_like(root_child)
            input_entities = input_tensor.data_subjects
        elif isinstance(input_tensor.data_subjects, DataSubjectList):
            input_entities = input_tensor.data_subjects.data_subjects_indexed
        else:
            raise NotImplementedError(
                f"Undefined behaviour for data subjects type: {type(input_tensor.data_subjects)}"
            )
            # data_subjects.data_subjects_indexed[0].reshape(-1)
            # t2 = time()
            # print("Obtained RDP Params, calculation time", t2 - t1)

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
            # print("Fixed mask shape!")
            # here we have the final mask and highest possible spend has been applied
            # to the data scientists budget field in the database

            if mask is None:
                raise Exception("Failed to publish mask is None")

            # print("Obtained overbudgeted entity mask", mask.dtype)

            # multiply values by the inverted mask
            filtered_input_tensor = value * (
                    1 - reshaped_mask
            )  # + gauss(0, sigma)  # Double check that noise has mean of 0

            filtered_inputs.append(filtered_input_tensor)
            # output = np.asarray(output_func(filtered_inputs) + noise)

        except Exception as e:
            # stdlib
            import traceback

            print(traceback.format_exc())
            print(f"Failed to run vectorized_publish. {e}")

    print("We have filtered all the input tensors. Now to compute the result:")

    # noise = secrets.SystemRandom().gauss(0, sigma)
    print(
        "Filtered inputs ",
        type(filtered_inputs),
        type(filtered_input_tensor),
        filtered_inputs,
    )

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
