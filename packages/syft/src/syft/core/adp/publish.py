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
    scalars: TypeList[Any], acc: Any, user_key: VerifyKey, sigma: float = 1.5
) -> TypeList[Any]:
    acc.temp_entity2ledger = {}

    ms = get_all_entity_mechanisms(scalars=scalars, sigma=sigma)

    # add the user_key to all of the mechanisms
    for _, mechs in ms.items():
        for m in mechs:
            m.user_key = user_key

    acc.temp_append(ms)

    overbudgeted_entities = acc.overbudgeted_entities(
        temp_entities=acc.temp_entity2ledger, user_key=user_key
    )

    # so that we don't modify the original polynomial
    # it might be fine to do so but just playing it safe
    if len(overbudgeted_entities) > 0:
        scalars = deepcopy(scalars)

    iterator = 0
    while len(overbudgeted_entities) > 0 and iterator < 3:
        print("\n\n QUERY IS OVER BUDGET!!! \n\n")

        iterator += 1

        input_scalars = set()
        for output_scalar in scalars:

            # output_scalar.input_scalars is a @property which determines
            # what inputs are still contributing to the output scalar
            # given that we may have just removed some
            for input_scalar in output_scalar.input_scalars:
                input_scalars.add(input_scalar)

        # should_break = False

        for input_scalar in input_scalars:
            if input_scalar.entity in overbudgeted_entities:
                for output_scalar in scalars:

                    # remove input_scalar from the computation that creates
                    # output scalar because this input_scalar is causing
                    # the budget spend to be too high.
                    output_scalar.poly = SubstitutionMapper(
                        make_subst_func({input_scalar.poly.name: 0})
                    )(output_scalar.poly)
            #
            # if should_break:
            #     break

        acc.temp_entity2ledger = {}

        # get mechanisms for new publish event
        ms = get_all_entity_mechanisms(scalars=scalars, sigma=sigma)

        for _, mechs in ms.items():
            for m in mechs:
                m.user_key = user_key

        # this is when we actually insert into the database
        acc.temp_append(ms)

        overbudgeted_entities = acc.overbudgeted_entities(
            temp_entities=acc.temp_entity2ledger, user_key=user_key
        )

    output = [s.value + random.gauss(0, sigma) for s in scalars]

    acc.save_temp_ledger_to_longterm_ledger()

    return output


def get_mechanism_for_entity(
    scalars: TypeList[Any], entity: Entity, sigma: float = 1.5
) -> Type[iDPGaussianMechanism]:

    m_id = "ms_"
    for s in scalars:
        m_id += str(s.id).split(" ")[1][:-1] + "_"

    value = np.sqrt(np.sum(np.square(np.array([float(s.value) for s in scalars]))))

    L = float(max_lipschitz_wrt_entity(scalars, entity=entity))

    return iDPGaussianMechanism(
        sigma=sigma,
        value=value,
        L=L,
        entity_name=entity.name,
        name=m_id,
    )


def get_all_entity_mechanisms(
    scalars: TypeList[Any], sigma: float = 1.5
) -> TypeDict[Entity, Any]:
    entities = set()
    for s in scalars:
        for i_s in s.input_scalars:
            entities.add(i_s.entity)
    return {
        e: [get_mechanism_for_entity(scalars=scalars, entity=e, sigma=sigma)]
        for e in entities
    }
