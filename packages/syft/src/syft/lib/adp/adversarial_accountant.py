# stdlib
from copy import deepcopy
import random
from typing import Dict as TypeDict
from typing import KeysView as TypeKeysView
from typing import List as TypeList
from typing import Set as TypeSet

# third party
from autodp.autodp_core import Mechanism
from autodp.transformer_zoo import Composition
import numpy as np

# syft absolute
import syft.lib.adp as adp

# syft relative
from .entity import Entity
from .idp_gaussian_mechanism import iDPGaussianMechanism
from .search import max_lipschitz_wrt_entity


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


def publish(scalars, acc, sigma: float = 1.5) -> float:
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


class AdversarialAccountant:
    def __init__(self, max_budget: float = 10, delta: float = 1e-6) -> None:
        self.entity2ledger: TypeDict[Entity, Mechanism] = {}
        self.max_budget = max_budget
        self.delta = delta

    def append(self, entity2mechanisms: TypeDict[str, TypeList[Mechanism]]) -> None:
        for key, ms in entity2mechanisms.items():
            if key not in self.entity2ledger.keys():
                self.entity2ledger[key] = list()
            for m in ms:
                self.entity2ledger[key].append(m)

    def get_eps_for_entity(self, entity: Entity) -> "Scalar":
        # compose them with the transformation: compose.
        compose = Composition()
        mechanisms = self.entity2ledger[entity]
        composed_mech = compose(mechanisms, [1] * len(mechanisms))

        # Query for eps given delta
        return adp.scalar.PhiScalar(
            value=composed_mech.get_approxDP(self.delta),
            min_val=0,
            max_val=self.max_budget,
            entity=entity,
        )

    def has_budget(self, entity_name: str) -> bool:
        eps = self.get_eps_for_entity(entity_name)
        if eps.value is not None:
            return eps.value < self.max_budget

    @property
    def entities(self) -> TypeKeysView[str]:
        return self.entity2ledger.keys()

    @property
    def overbudgeted_entities(self) -> TypeSet[str]:
        entities = set()

        for ent in self.entities:
            if not self.has_budget(ent):
                entities.add(ent)

        return entities

    def print_ledger(self, delta: float = 1e-6) -> None:
        for entity, mechanisms in self.entity2ledger.items():
            print(str(entity) + "\t" + str(self.get_eps_for_entity(entity)._value))
