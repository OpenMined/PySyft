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
from typing import List
from typing import List as TypeList
from typing import Tuple
from typing import Union

# third party
from nacl.signing import VerifyKey
import numpy as np
from pymbolic.mapper.substitutor import SubstitutionMapper
from pymbolic.mapper.substitutor import make_subst_func

# relative
from .entity import DataSubjectGroup
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
    print("publish.py:45: TRY: ms = get_all_entity_mechanisms")
    # Recover all entity mechanisms
    ms = get_all_entity_mechanisms(
        scalars=scalars, sigma=sigma, public_only=public_only
    )
    print("publish.py:50: SUCCESS: ms = get_all_entity_mechanisms")

    print("publish.py:52: TRY: for _, mechs in ms.items():")
    # add the user_key to all of the mechanisms
    for _, mechs in ms.items():
        for m in mechs:
            m.user_key = user_key
    print("publish.py:57: SUCCESS: for _, mechs in ms.items():")

    print("publish.py:59: TRY: acc.temp_append(ms)")
    # Register the mechanism / entities at the temporary data structure
    # This data structure will be organized as a dictionary of
    # lists, each list will contain a set of mechanisms related to an entity.
    # Example: acc.temp_entity2ledger = {"patient_1": [<iDPGaussianMechanism>, <iDPGaussianMechanism>] }
    acc.temp_append(ms)
    print("publish.py:65: SUCCESS: acc.temp_append(ms)")

    print("publish.py:67: TRY: overbudgeted_entities = acc.overbudgeted_entities(")
    # Filter entities by searching for the overbudgeted ones.
    overbudgeted_entities = acc.overbudgeted_entities(
        temp_entities=acc.temp_entity2ledger,
        user_key=user_key,
        returned_epsilon_is_private=True,
    )
    print("publish.py:74: SUCCESS: overbudgeted_entities = acc.overbudgeted_entities(")

    print(
        "publish.py:76: TRY:  if len(overbudgeted_entities) > 0: scalars = deepcopy(scalars)"
    )
    # so that we don't modify the original polynomial
    # it might be fine to do so but just playing it safe
    if len(overbudgeted_entities) > 0:
        scalars = deepcopy(scalars)
    print(
        "publish.py:81: SUCCESS:  if len(overbudgeted_entities) > 0: scalars = deepcopy(scalars)"
    )

    # If some overbudgeted entity is found, run this.
    iterator = 0
    while len(overbudgeted_entities) > 0 and iterator < 3:
        print(
            "publish.py:86: INSIDE:  while len(overbudgeted_entities) > 0 and iterator < 3:"
        )
        iterator += 1
        print(
            "publish.py:88: len(overbudgeted_entities) == "
            + str(len(overbudgeted_entities))
        )
        print("publish.py:89:  for output_scalar in scalars:")
        input_scalars = set()
        for output_scalar in scalars:

            print("publish.py:93:  for input_scalar in output_scalar.input_scalars:")
            # output_scalar.input_scalars is a @property which determines
            # what inputs are still contributing to the output scalar
            # given that we may have just removed some
            for input_scalar in output_scalar.input_scalars:
                input_scalars.add(input_scalar)
        print("publish.py:99:  for input_scalar in input_scalars:")
        for input_scalar in input_scalars:
            if input_scalar.entity in overbudgeted_entities:
                for output_scalar in scalars:

                    # remove input_scalar from the computation that creates
                    # output scalar because this input_scalar is causing
                    # the budget spend to be too high.
                    output_scalar.poly = SubstitutionMapper(
                        make_subst_func({input_scalar.poly.name: 0})
                    )(output_scalar.poly)

        print("publish.py:110:  acc.temp_entity2ledger = {}")
        acc.temp_entity2ledger = {}

        print("publish.py:114:  BEGIN: ms = get_all_entity_mechanisms(")
        # get mechanisms for new publish event
        ms = get_all_entity_mechanisms(
            scalars=scalars, sigma=sigma, public_only=public_only
        )
        print("publish.py:119:  SUCCESS: ms = get_all_entity_mechanisms(")

        for _, mechs in ms.items():
            for m in mechs:
                m.user_key = user_key

        print("publish.py:127:  TRY: acc.temp_append(ms)")
        # this is when we actually insert into the database
        acc.temp_append(ms)
        print("publish.py:127:  SUCCESS: acc.temp_append(ms)")

        print(
            "publish.py:130:  TRY:  overbudgeted_entities = acc.overbudgeted_entities("
        )
        overbudgeted_entities = acc.overbudgeted_entities(
            temp_entities=acc.temp_entity2ledger,
            user_key=user_key,
            returned_epsilon_is_private=True,
        )
        print(
            "publish.py:136:  SUCCESS:  overbudgeted_entities = acc.overbudgeted_entities("
        )

    print(
        "publish.py:138:  TRY:  output = [s.value + random.gauss(0, sigma) for s in scalars]"
    )
    # Add to each scalar a a gaussian noise in an interval between
    # 0 to sigma value.
    output = [s.value + random.gauss(0, sigma) for s in scalars]
    print(
        "publish.py:142:  SUCCESS:  output = [s.value + random.gauss(0, sigma) for s in scalars]"
    )

    print("publish.py:144:  TRY:  acc.save_temp_ledger_to_longterm_ledger()")
    # Persist the temporary ledger into the database.
    acc.save_temp_ledger_to_longterm_ledger()
    print("publish.py:147:  SUCCESS:  acc.save_temp_ledger_to_longterm_ledger()")

    return output


def get_mechanism_for_entity(
    scalars: TypeList[Any],
    entity: Union[Entity, DataSubjectGroup],
    sigma: float = 1.5,
    public_only: bool = False,
) -> Union[
    List[Tuple[Entity, iDPGaussianMechanism]],
    List[Tuple[DataSubjectGroup, iDPGaussianMechanism]],
]:
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

    if isinstance(entity, DataSubjectGroup):
        mechanisms = []

        for e in entity.entity_set:
            mechanisms.append(
                (
                    entity,
                    iDPGaussianMechanism(
                        sigma=sigma,
                        squared_l2_norm=value,
                        squared_l2_norm_upper_bound=value_upper_bound,
                        L=float(max_lipschitz_wrt_entity(scalars, entity=e)),
                        entity_name=e.name,
                        name=m_id,
                    ),
                )
            )
        return mechanisms
    elif isinstance(entity, Entity):
        return [
            (
                entity,
                iDPGaussianMechanism(
                    sigma=sigma,
                    squared_l2_norm=value,
                    squared_l2_norm_upper_bound=value_upper_bound,
                    L=float(max_lipschitz_wrt_entity(scalars, entity=entity)),
                    entity_name=entity.name,
                    name=m_id,
                ),
            )
        ]
    else:
        raise Exception(f"What even is this type: {type(entity)}")


def get_all_entity_mechanisms(
    scalars: TypeList[Any], sigma: float = 1.5, public_only: bool = False
) -> TypeDict[Entity, Any]:
    "Generates a list of entities by processing the given scalars list."
    entities = set()
    for s in scalars:
        for i_s in s.input_scalars:
            entities.add(i_s.entity)
    entity_to_mechanisms: dict = {}
    for entity in entities:
        for flat_entity, mechanism in get_mechanism_for_entity(
            scalars=scalars, entity=entity, sigma=sigma, public_only=public_only
        ):
            # We ignore entity in this case because entity might be a DataSubjectGroup,
            # in which case flat_entity will include more entities than entity
            if flat_entity not in entity_to_mechanisms:
                entity_to_mechanisms[flat_entity] = list()
            entity_to_mechanisms[flat_entity].append(mechanism)
    return entity_to_mechanisms


def get_remaining_budget(acc: Any, user_key: VerifyKey) -> Any:
    budget = acc.get_remaining_budget(
        user_key=user_key, returned_epsilon_is_private=False
    )
    return budget
