# CLEANUP NOTES:
# - remove unused comments
# - add documentation for each method
# - add comments inline explaining each piece
# - add a unit test for each method (at least)

# stdlib
from copy import deepcopy
import random
from typing import Any
from typing import Dict as TypeDict
from typing import List as TypeList
from typing import Type

# third party
from nacl.signing import VerifyKey
import numpy as np
from pymbolic.mapper.substitutor import SubstitutionMapper
from pymbolic.mapper.substitutor import make_subst_func

# relative
from .entity import Entity
from .idp_gaussian_mechanism import iDPGaussianMechanism
from .search import max_lipschitz_wrt_entity


def publish(
    scalars: TypeList[Any],
    acc: Any,
    user_key: VerifyKey,
    sigma: float = 1.5,
    public_only: bool = False,
) -> TypeList[Any]:
    """
    Method to compute and update the mechanism values of data entities managed by the Adversarial Accountant
    class.
    """

    # Creates a temporary register (memory) to
    # store entities and their respective mechanisms
    acc.temp_entity2ledger = {}

    # Recover all entity mechanisms
    ms = get_all_entity_mechanisms(
        scalars=scalars, sigma=sigma, public_only=public_only
    )

    # add the user_key to all of the mechanisms
    for _, mechs in ms.items():
        for m in mechs:
            m.user_key = user_key

    # Register the mechanism / entities at the temporary data structure
    # This data structure will be organized as a dictionary of
    # lists, each list will contain a set of mechanisms related to an entity.
    # Example: acc.temp_entity2ledger = {"patient_1": [<iDPGaussianMechanism>, <iDPGaussianMechanism>] }
    acc.temp_append(ms)

    # Filter entities by searching for the overbudgeted ones.
    overbudgeted_entities = acc.overbudgeted_entities(
        temp_entities=acc.temp_entity2ledger,
        user_key=user_key,
        returned_epsilon_is_private=True,
    )

    # so that we don't modify the original polynomial
    # it might be fine to do so but just playing it safe
    if len(overbudgeted_entities) > 0:
        scalars = deepcopy(scalars)

    # If some overbudgeted entity is found, run this.
    iterator = 0
    while len(overbudgeted_entities) > 0 and iterator < 3:

        iterator += 1

        input_scalars = set()
        for output_scalar in scalars:

            # output_scalar.input_scalars is a @property which determines
            # what inputs are still contributing to the output scalar
            # given that we may have just removed some
            for input_scalar in output_scalar.input_scalars:
                input_scalars.add(input_scalar)

        for input_scalar in input_scalars:
            if input_scalar.entity in overbudgeted_entities:
                for output_scalar in scalars:

                    # remove input_scalar from the computation that creates
                    # output scalar because this input_scalar is causing
                    # the budget spend to be too high.
                    output_scalar.poly = SubstitutionMapper(
                        make_subst_func({input_scalar.poly.name: 0})
                    )(output_scalar.poly)

        acc.temp_entity2ledger = {}

        # get mechanisms for new publish event
        ms = get_all_entity_mechanisms(
            scalars=scalars, sigma=sigma, public_only=public_only
        )

        for _, mechs in ms.items():
            for m in mechs:
                m.user_key = user_key

        # this is when we actually insert into the database
        acc.temp_append(ms)

        overbudgeted_entities = acc.overbudgeted_entities(
            temp_entities=acc.temp_entity2ledger,
            user_key=user_key,
            returned_epsilon_is_private=True,
        )

    # Add to each scalar a a gaussian noise in an interval between
    # 0 to sigma value.
    output = [s.value + random.gauss(0, sigma) for s in scalars]

    # Persist the temporary ledger into the database.
    acc.save_temp_ledger_to_longterm_ledger()

    return output


def get_mechanism_for_entity(
    scalars: TypeList[Any],
    entity: Entity,
    sigma: float = 1.5,
    public_only: bool = False,
) -> Type[iDPGaussianMechanism]:
    """
    Iterates over scalars computing its value and L attribute and builds its mechanism.
    """
    m_id = "ms_"
    for s in scalars:
        m_id += str(s.id).split(" ")[1][:-1] + "_"

    value_upper_bound = np.sqrt(
        np.sum(
            np.square(
                np.array([(float(s.max_val) - float(s.min_val)) for s in scalars])
            )
        )
    )

    if public_only:
        value = value_upper_bound
    else:
        value = np.sqrt(np.sum(np.square(np.array([float(s.value) for s in scalars]))))

    L = float(max_lipschitz_wrt_entity(scalars, entity=entity))

    return iDPGaussianMechanism(
        sigma=sigma,
        squared_l2_norm=value,
        squared_l2_norm_upper_bound=value_upper_bound,
        L=L,
        entity_name=entity.name,
        name=m_id,
    )


def get_all_entity_mechanisms(
    scalars: TypeList[Any], sigma: float = 1.5, public_only: bool = False
) -> TypeDict[Entity, Any]:
    "Generates a list of entities by processing the given scalars list."
    entities = set()
    for s in scalars:
        for i_s in s.input_scalars:
            entities.add(i_s.entity)
    return {
        e: [
            get_mechanism_for_entity(
                scalars=scalars, entity=e, sigma=sigma, public_only=public_only
            )
        ]
        for e in entities
    }


def get_remaining_budget(acc: Any, user_key: VerifyKey) -> Any:
    budget = acc.get_remaining_budget(
        user_key=user_key, returned_epsilon_is_private=False
    )
    return budget
