# stdlib
from typing import Dict as TypeDict
from typing import KeysView as TypeKeysView
from typing import List as TypeList
from typing import Set as TypeSet

# third party
from autodp.autodp_core import Mechanism
from autodp.transformer_zoo import Composition
from sqlalchemy.engine import Engine


# relative
from .entity import Entity
from .scalar import PhiScalar
from ..node.common.node_manager.ledger_manager import LedgerManager
from ..common.serde.recursive import RecursiveSerde

class AdversarialAccountant:
    def __init__(self, db_engine: Engine, max_budget: float = 10, delta: float = 1e-6) -> None:
        self.entity2ledger = LedgerManager(db_engine)
        self.max_budget = max_budget
        self.delta = delta

    def append(self, entity2mechanisms: TypeDict[Entity, TypeList[Mechanism]]) -> None:
        for key, ms in entity2mechanisms.items():
            if key not in self.entity2ledger.keys():
                self.entity2ledger[key] = list()
            for m in ms:
                self.entity2ledger.append(key, m)

    def get_eps_for_entity(self, entity_name: str) -> PhiScalar:
        # compose them with the transformation: compose
        compose = Composition()
        mechanisms = self.entity2ledger[entity_name]
        composed_mech = compose(mechanisms, [1] * len(mechanisms))

        return composed_mech.get_approxDP(self.delta)

        # # Query for eps given delta
        # return PhiScalar(
        #     value=composed_mech.get_approxDP(self.delta),
        #     min_val=0,
        #     max_val=self.max_budget,
        #     entity=entity,
        # )

    def has_budget(self, entity_name: Entity) -> bool:
        eps = self.get_eps_for_entity(entity_name)
        if eps is not None:
            return eps < self.max_budget
        # if eps.value is not None:
        #     return eps.value < self.max_budget

    @property
    def entities(self) -> TypeKeysView[Entity]:
        return self.entity2ledger.keys()

    @property
    def overbudgeted_entities(self) -> TypeSet[Entity]:
        entities = set()

        for entity_name in self.entities:
            if not self.has_budget(entity_name):
                entities.add(entity_name)

        return entities

    def print_ledger(self, delta: float = 1e-6) -> None:
        for entity, mechanisms in self.entity2ledger.items():
            print(str(entity) + "\t" + str(self.get_eps_for_entity(entity)))

class AccountantReference(RecursiveSerde):
    __attr_allowlist__ = ['msg']

    def __init__(self, msg):
        self.msg = msg
