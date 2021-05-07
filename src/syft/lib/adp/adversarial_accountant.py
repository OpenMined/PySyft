# stdlib
from typing import Dict as TypeDict
from typing import KeysView as TypeKeysView
from typing import List as TypeList
from typing import Set as TypeSet

# third party
from autodp.autodp_core import Mechanism
from autodp.transformer_zoo import Composition

# syft relative
from .scalar import Scalar


class AdversarialAccountant:
    def __init__(self, max_budget: float = 10, delta: float = 1e-6) -> None:
        self.entity2ledger: TypeDict[str, Mechanism] = {}
        self.max_budget = max_budget
        self.delta = delta

    def append(self, entity2mechanisms: TypeDict[str, TypeList[Mechanism]]) -> None:
        for key, ms in entity2mechanisms.items():
            if key not in self.entity2ledger.keys():
                self.entity2ledger[key] = list()
            for m in ms:
                self.entity2ledger[key].append(m)

    def get_eps_for_entity(self, entity_name: str) -> Scalar:
        # compose them with the transformation: compose.
        compose = Composition()
        mechanisms = self.entity2ledger[entity_name]
        composed_mech = compose(mechanisms, [1] * len(mechanisms))

        # Query for eps given delta
        return Scalar(
            value=composed_mech.get_approxDP(self.delta),
            min_val=0,
            max_val=self.max_budget,
            entity=entity_name,
        )

    def has_budget(self, entity_name: str) -> bool:
        eps = self.get_eps_for_entity(entity_name)
        value = eps.value
        if value is not None:
            return value < self.max_budget
        return True

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
        for entity_name, mechanisms in self.entity2ledger.items():
            print(entity_name + "\t" + str(self.get_eps_for_entity(entity_name)._value))
