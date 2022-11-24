# future
from __future__ import annotations

# stdlib
from collections.abc import Iterable
from copy import deepcopy
import secrets
from typing import Callable
from typing import TYPE_CHECKING
from typing import Tuple

# third party
import jax
from jax import numpy as jnp
import numpy as np
from numpy.typing import ArrayLike

# relative
from ...core.node.common.node_manager.user_manager import RefreshBudgetException
from ...core.tensor.autodp.gamma_tensor_ops import GAMMA_TENSOR_OP
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
    private: bool = True,
) -> np.ndarray:
    """
    This method applies Individual Differential Privacy (IDP) as defined in
    https://arxiv.org/abs/2008.11193
        - Key results: Theorem 2.7 and 2.8 show how much privacy budget is spent by a query.

    Given a tensor, it checks if the user (a data scientist) has enough privacy budget (PB)
    to see data from every data subject.
    - If the user has enough privacy budget, then DP noise is directly added to the result,
      and PB is deducted.
    - If the user doesn't have enough PB, then every data subject with a higher epsilon
      than their PB's data is removed.
        - The epsilons are then recomputed to see if the user now has enough PB to see the
          remaining data or not.

    Notes:
        - The privacy budget spent by a query equals the maximum epsilon increase of any
          data subject in the dataset.
    """
    # Step 0: Ensure our Tensor's private data is in a form that is usable.
    if isinstance(tensor.child, FixedPrecisionTensor):
        # Incase SMPC is involved, there will be an FPT in the chain to account for
        value = tensor.child.decode()
    else:
        value = tensor.child

    while isinstance(value, PassthroughTensor):
        # TODO: Ask Rasswanth why this check is necessary
        # ATTENTION: we do the same unboxing below with root_child
        # is this still needed to be done twice?
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

    # its important that its the same type so that eq comparisons below dont break
    zeros_like = jnp.zeros_like(value)

    # this prevents us from running in an infinite loop
    previous_budget = None
    previous_spend = None

    # if we dont return below we will terminate if the tensor gets replaced with zeros
    prev_tensor = None

    while can_reduce_further(value=value, zeros_like=zeros_like):
        if prev_tensor is None:
            prev_tensor = value
        else:
            if (prev_tensor == value).all():  # type: ignore
                raise Exception("Tensor has not changed and is not all zeros")
            else:
                prev_tensor = value

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
        rdp_constants = compute_rdp_constant(rdp_params, private=private)
        if any(rdp_constants < 0):
            raise Exception(
                "Negative budget spend not allowed in PySyft for safety reasons."
                "Please contact the OpenMined support team for help."
                "For that you can either:"
                " * describe your issue on our Slack #support channel. To join: https://openmined.slack.com/"
                " * send us an email describing your problem at support@openmined.org"
                " * leave us an issue here: https://github.com/OpenMined/PySyft/issues"
            )
        if any(np.isnan(rdp_constants)) or any(np.isinf(rdp_constants)):
            raise Exception(
                "Invalid privacy budget spend. Please contact the OpenMined support team for help."
                "For that you can either:"
                " * describe your issue on our Slack #support channel. To join: https://openmined.slack.com/"
                " * send us an email describing your problem at support@openmined.org"
                " * leave us an issue here: https://github.com/OpenMined/PySyft/issues"
            )
        all_epsilons = ledger._get_epsilon_spend(
            rdp_constants
        )  # This is the epsilon spend for ALL data subjects
        if any(all_epsilons < 0):
            raise Exception(
                "Negative budget spend not allowed in PySyft for safety reasons."
                "Please contact the OpenMined support team for help."
                "For that you can either:"
                " * describe your issue on our Slack #support channel. To join: https://openmined.slack.com/"
                " * send us an email describing your problem at support@openmined.org"
                " * leave us an issue here: https://github.com/OpenMined/PySyft/issues"
            )

        epsilon_spend = max(
            all_epsilons
        )  # This is the epsilon spend for the QUERY, a single float.

        if not isinstance(epsilon_spend, float):
            epsilon_spend = float(epsilon_spend)

        if epsilon_spend < 0:
            raise Exception(
                "Negative budget spend not allowed in PySyft for safety reasons."
                "Please contact the OpenMined support team for help."
                "For that you can either:"
                " * describe your issue on our Slack #support channel. To join: https://openmined.slack.com/"
                " * send us an email describing your problem at support@openmined.org"
                " * leave us an issue here: https://github.com/OpenMined/PySyft/issues"
            )

        # Step 3: Check if the user has enough privacy budget for this query
        privacy_budget = get_budget_for_user(verify_key=ledger.user_key)
        has_budget = epsilon_spend <= privacy_budget

        # if we see the same budget and spend twice in a row we have failed to reduce it
        if (
            privacy_budget == previous_budget
            and epsilon_spend == previous_spend
            and not has_budget
        ):
            raise Exception(
                "Publish has failed to reduce spend. "
                f"With Budget: {previous_budget} Spend: {epsilon_spend}. Aborting."
            )

        if previous_budget is None:
            previous_budget = privacy_budget

        if previous_spend is None:
            previous_spend = epsilon_spend

        # Step 4: Path 1 - If the User has enough Privacy Budget, we just add noise,
        # deduct budget, and return the result.
        if has_budget:
            original_output = value

            # We sample noise from a cryptographically secure distribution
            # TODO: Replace with discrete gaussian distribution instead of regular
            # gaussian to eliminate floating pt vulns
            noise = np.asarray(
                [
                    secrets.SystemRandom().gauss(0, sigma)
                    for _ in range(original_output.size)
                ]
            ).reshape(original_output.shape)

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
            ledger.update_rdp_constants(
                query_constants=rdp_constants, entity_ids_query=input_entities
            )
            ledger._write_ledger()
            return original_output + noise

        # Step 4: Path 2 - User doesn't have enough privacy budget.
        elif not has_budget:
            print("Not enough privacy budget, about to start filtering.")
            # If the user doesn't have enough PB, they shouldn't see data of high
            # epsilon data subjects (privacy violation)
            # So we will remove data belonging to these data subjects from the computation.

            # Step 4.1: Figure out which data subjects are within the PB & the highest
            # possible spend
            # within_budget_filter = (
            #     jnp.ones_like(all_epsilons) * privacy_budget >= all_epsilons
            # )
            # highest_possible_spend = jnp.max(all_epsilons * within_budget_filter)
            # TODO: Modify to work with private/public operations (when input_tensor is a scalar)

            # Step 4.2: Figure out which Tensors in the Source dictionary have those data subjects

            # create a seperate iterable of the keys so they can be mutated below
            filtered_sourcetree = deepcopy(tensor.sources)
            input_tensors = list(filtered_sourcetree.values())

            parent_branch = [
                filtered_sourcetree for _ in input_tensors
            ]  # TODO: Ensure this isn't deepcopying!

            # relative
            from ..tensor.autodp.gamma_tensor import GammaTensor

            # ATTENTION: is this the same as tensor.sources.items() ?
            for parent_state, input_tensor in zip(parent_branch, input_tensors):

                if isinstance(input_tensor, GammaTensor):
                    if (
                        input_tensor.func_str == GAMMA_TENSOR_OP.NOOP.value
                    ):  # This is raw, unprocessed private data. Filter if eps spend > PB!
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
                        epsilon = max(
                            ledger._get_epsilon_spend(
                                np.asarray(
                                    compute_rdp_constant(rdp_params, private=private)
                                )
                            )
                        )

                        # Filter if > privacy budget
                        if jnp.isnan(epsilon):
                            raise Exception("Epsilon is NaN")

                        if epsilon > privacy_budget:
                            filtered_tensor = input_tensor.filtered()

                            # Replace the original tensor with this filtered one
                            # remove the original state id

                            # TODO: This changes insertion order which could impact
                            # the non commutable operations like div where by the
                            # values are unpacked in order of insertion before being
                            # executed as a tuple
                            # see def _truediv(state: dict) -> jax.numpy.DeviceArray:
                            # in gamma_functions.py

                            del parent_state[input_tensor.id]

                            # add the new zeroed state id tensor
                            parent_state[filtered_tensor.id] = filtered_tensor
                        # If epsilon <= privacy budget, we don't need to do anything -
                        # the user has enough PB to use the data
                    else:
                        # Is this supposed to search the entire state tree with no_op
                        # data nodes being the leaves?

                        # ATTENTION: is this intended to walk the state tree?
                        # now that we no longer use recursion, I don't think this will
                        # work so we might want to have a stack / queue above to
                        # explore the frontier
                        input_tensors += list(input_tensor.sources.values())
                        parent_branch += [
                            input_tensor.sources for _ in input_tensor.sources.values()
                        ]
                else:
                    # This is a public value, we don't touch 'em.
                    continue

            # Recompute the tensor's value now that some of its inputs have been
            # filtered, and repeat epsilon calculations

            # ATTENTION: When we swap state we need to handle the base case so that
            # the .child gets replaced not the .source (state) otherwise it never
            # terminates
            tensor = tensor.swap_state(filtered_sourcetree)
            print("tensor.child before restart: ", type(tensor.child), tensor.child)
            print("About to publish again with filtered source_tree!")

            # TODO: This isn't the most efficient way to do it since we can reuse sigmas, coeffs, etc.
            # TODO: Add a way to prevent infinite publishing?
            # TODO: Should we implement exponential backoff or something as a means of rate-limiting?
            # return new_tensor.publish(
            #     get_budget_for_user=get_budget_for_user,
            #     deduct_epsilon_for_user=deduct_epsilon_for_user,
            #     ledger=ledger,
            #     sigma=sigma,
            # )
        # Step 5: Revel in happiness.

        else:
            raise Exception

    noise = np.asarray(
        [secrets.SystemRandom().gauss(0, sigma) for _ in range(zeros_like.size)]
    ).reshape(zeros_like.shape)
    zeros = np.zeros_like(
        a=np.array([]), dtype=zeros_like.dtype, shape=zeros_like.shape
    )
    return zeros + noise


# each time we attempt to publish and filter values we need to ensure
# the while loop comparison is valid. We check for three edge cases.
# 1) the result of comparison is a non scalar because scalars cant be reduced
# 2) there are differences between the value and a zeros_like of the same shape
# 3) within the value ArrayLike there are no NaN values as these will never evaluate
# to True when compared with zeros_like and therefore never exit the loop
def can_reduce_further(value: ArrayLike, zeros_like: ArrayLike) -> bool:
    try:
        result = value != zeros_like
        # check we can call any or iterate on this value otherwise exit loop
        # numpy scalar types like np.bool_ are Iterable
        if not hasattr(result, "any") and not isinstance(result, Iterable):
            return False

        # make sure the comparison has some difference and there are also no NaNs
        # causing that difference in the result
        return result.any() and not_nans(result)
    except Exception as e:
        print(f"Unable to test reducability of {type(value)} and {type(zeros_like)}")
        raise e


# there are some types which numpy will not allow an isnan check such as strings
# so we should be extra careful to notify the user of why this check failed
def not_nans(value: ArrayLike) -> bool:
    try:
        # TODO: support nan ufunc properly?
        # do we need to add _array attrs to our tensor chain?
        while hasattr(value, "child"):
            value = value.child

        if isinstance(value, np.ndarray):
            return not np.isnan(value).any()
        else:
            return not jnp.isnan(value).any()
    except Exception as e:
        print(f"Holy Batmanananana. isnan is not supported {type(value)}.")
        raise e
