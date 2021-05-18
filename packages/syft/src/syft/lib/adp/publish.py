# stdlib
from copy import deepcopy
import random
from typing import Any

# third party
import numpy as np

# syft relative
from .idp_gaussian_mechanism import iDPGaussianMechanism
from .search import max_lipschitz_wrt_entity


def publish(scalars, acc: Any, sigma: float = 1.5) -> float:
    acc_original = acc

    acc_temp = deepcopy(acc_original)

    ms = get_all_entity_mechanisms(scalars=scalars, sigma=sigma)
    acc_temp.append(ms)

    overbudgeted_entities = acc_temp.overbudgeted_entities

    # so that we don't modify the original polynomial
    # it might be fine to do so but just playing it safe
    if len(overbudgeted_entities) > 0:
        scalars = deepcopy(scalars)

    while len(overbudgeted_entities) > 0:

        input_scalars = set()
        for s in scalars:
            for i_s in s.input_scalars:
                input_scalars.add(i_s)

        for symbol in input_scalars:
            if symbol.entity in overbudgeted_entities:
                symbol.poly = symbol.poly.subs(symbol.poly, 0)
                break

        acc_temp = deepcopy(acc_original)

        # get mechanisms for new publish event
        ms = get_all_entity_mechanisms(scalars=scalars, sigma=sigma)
        acc_temp.append(ms)

        overbudgeted_entities = acc_temp.overbudgeted_entities

    output = [s.value + random.gauss(0, sigma) for s in scalars]

    acc_original.entity2ledger = deepcopy(acc_temp.entity2ledger)

    return output


def get_mechanism_for_entity(scalars, entity, sigma=1.5):
    m_id = "ms_"
    for s in scalars:
        m_id += str(s.id).split(" ")[1][:-1] + "_"

    return iDPGaussianMechanism(
        sigma=sigma,
        value=np.sqrt(np.sum(np.square(np.array([float(s.value) for s in scalars])))),
        L=float(max_lipschitz_wrt_entity(scalars, entity=entity)),
        entity=entity.unique_name,
        name=m_id,
    )


def get_all_entity_mechanisms(scalars, sigma: float = 1.5):
    entities = set()
    for s in scalars:
        for i_s in s.input_scalars:
            entities.add(i_s.entity)
    return {
        e: [get_mechanism_for_entity(scalars=scalars, entity=e, sigma=sigma)]
        for e in entities
    }
