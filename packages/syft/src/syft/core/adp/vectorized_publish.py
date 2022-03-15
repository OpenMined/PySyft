
# stdlib
from copy import deepcopy
import random
from typing import Any
from typing import Dict as TypeDict
from typing import List
from typing import List as TypeList
from typing import Tuple
from typing import Union

# third party
from nacl.signing import VerifyKey
from jax import numpy as jnp
import numpy as np

# relative
from .entity import DataSubjectGroup
from .entity import Entity
from .entity_list import EntityList
from .adversarial_accountant import AdversarialAccountant
from .idp_gaussian_mechanism import iDPGaussianMechanism
from .search import max_lipschitz_wrt_entity


def publish(values: jnp.array,
            min_vals: float,
            max_val: float,
            data_subjects: EntityList,
            is_linear: bool,
            acc: AdversarialAccountant,
            user_key: VerifyKey,
            sigma: float = 1.5,
            public_only: bool = False,
            ) -> jnp.array:
    # Step 1: Get all mechanisms for each entity/data subject
    unique_data_subjects = data_subjects.one_hot_lookup  # this is an array of all unique data subjects

    squared_l2_norm, worst_case_squared_l2_norm = calculate_bounds_for_mechanism(values, min_vals, max_val, public_only)
    print(squared_l2_norm)
    print(worst_case_squared_l2_norm)

    if is_linear:
        lipschitz_array = np.ones_like(values)  # max lipschitz bound
    else:
        # Technically this is implemented (we call GammaTensor.lipschitz_bound) but let's deal with that later
        raise NotImplementedError

    return jnp.ones_like(values)  # temporary placeholder


def calculate_bounds_for_mechanism(value_array, min_val_array, max_val_array, public_only: bool):
    """Calculates the squared L2 norm values needed to create a Mechanism, and calculate privacy budget + spend"""
    """ If you calculate the privacy budget spend with the worst case bound, you can show this number to the D.S.
    If you calculate it with the regular value (the value computed below when public_only = False, you cannot show the 
    privacy budget to the DS because this violates privacy.
    """

    # TODO: Double check whether the iDPGaussianMechanism class squares its squared_l2_norm values!!
    worst_case_squared_l2_norm = np.sum(np.square(max_val_array - min_val_array))

    if public_only:
        squared_l2_norm = worst_case_squared_l2_norm
    else:
        squared_l2_norm = np.sum(np.square(value_array))

    return squared_l2_norm, worst_case_squared_l2_norm


def get_mechanism_for_unique_data_subjects(unique_ds: jnp.array):
    """
    For each data subject in the data_subjects array, we need to:
    1. Create some kind of mechanism ID
    - I think we can use the GammaTensor.id attribute (if we make the random number generation there secure)
    - Alternatively maybe we can use indices somehow
    """

    pass