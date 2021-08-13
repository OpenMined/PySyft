# stdlib
from typing import Dict as TypeDict
from typing import KeysView as TypeKeysView
from typing import List as TypeList
from typing import Optional
from typing import Set as TypeSet

# third party
from autodp.autodp_core import Mechanism
from autodp.transformer_zoo import Composition
from nacl.encoding import HexEncoder
from nacl.signing import VerifyKey
from sqlalchemy.engine import Engine

# relative
from syft.core.common.serde import _deserialize

from ..common.serde.recursive import RecursiveSerde
from ..node.common.node_manager.ledger_manager import LedgerManager
from .entity import Entity
from .scalar import PhiScalar


class AdversarialAccountant:
    def __init__(
        self, db_engine: Engine, max_budget: float = 10, delta: float = 1e-6
    ) -> None:
        self.entity2ledger = LedgerManager(db_engine)
        self.temp_entity2ledger = {}
        self.max_budget = max_budget
        self.delta = delta

    def temp_append(
        self, entity2mechanisms: TypeDict[Entity, TypeList[Mechanism]]
    ) -> None:
        for key, ms in entity2mechanisms.items():
            if key not in self.temp_entity2ledger.keys():
                self.temp_entity2ledger[key.name] = list()
            for m in ms:
                self.temp_entity2ledger[key.name].append(m)

    def append(self, entity2mechanisms: TypeDict[str, TypeList[Mechanism]]) -> None:

        mechanisms = list()
        for key, ms in entity2mechanisms.items():
            for m in ms:
                mechanisms.append(m)

        self.entity2ledger.register_mechanisms(mechanisms)


    def save_temp_ledger_to_longterm_ledger(self):
        self.append(entity2mechanisms=self.temp_entity2ledger)

    def get_eps_for_entity(
        self, entity_name: str, user_key: Optional[VerifyKey] = None
    ) -> PhiScalar:

        print("GET EPS FOR ENTITY()")

        # compose them with the transformation: compose
        compose = Composition()

        print("Entity Type:" + str(type(entity_name)))
        print("Entity:" + str(entity_name))
        print("entity name:" + str(entity_name.name))

        all_ents = self.entity2ledger.entity_manager.all()
        # for e in all_ents:
        #     print("Known Entity Name:" + str(e.name))

        # fetch mechanisms from the database
        table_mechanisms = self.entity2ledger.query(entity_name=entity_name.name)
        mechanisms = [x.obj for x in table_mechanisms]

        # print('Mechansms returned:' + str(mechanisms))

        # filter out mechanisms that weren't created by this data scientist user
        if user_key is not None:
            filtered_mechanisms = []
            for mech in mechanisms:
                # left = _deserialize(mech.user_key, from_bytes=True)
                # print("Comparing Left:" + str(mech.user_key) + " of type " + str(type(mech.user_key)))
                # print("Comparing Right:" + str(user_key) + " of type " + str(type(user_key)))

                # user_key = VerifyKey(user_key.encode("utf-8"), encoder=HexEncoder)


                if mech.user_key == user_key:
                    filtered_mechanisms.append(mech)

            mechanisms = filtered_mechanisms

        print("Num mechanisms before TEMP:" + str(len(mechanisms)))
        print("self.temp_entity2ledger:" + str(self.temp_entity2ledger))

        if entity_name.name in self.temp_entity2ledger.keys():
            mechanisms = mechanisms + self.temp_entity2ledger[entity_name.name]

        # filter out mechanisms that weren't created by this data scientist user
        if user_key is not None:
            filtered_mechanisms = []
            for mech in mechanisms:
                # left = _deserialize(mech.user_key, from_bytes=True)
                # print("Comparing Left:" + str(mech.user_key) + " of type " + str(type(mech.user_key)))
                # print("Comparing Right:" + str(user_key) + " of type " + str(type(user_key)))

                # user_key = VerifyKey(user_key.encode("utf-8"), encoder=HexEncoder)


                if mech.user_key == user_key:
                    filtered_mechanisms.append(mech)

            mechanisms = filtered_mechanisms

        print("Num mechanisms after TEMP:" + str(len(mechanisms)))
        # for m in mechanisms:
            # print("Filtered Mechanism Entity:" + str(m.entity))

        # print("Mechanisma after filtering: ", mechanisms)
        # use verify key to specify the user
        # for all entities in the db,
        # how do we ensure that no data scientist
        # exceeds the budget of any entity?

        # map dataset
        if len(mechanisms) > 0:
            composed_mech = compose(mechanisms, [1] * len(mechanisms))
            eps = composed_mech.get_approxDP(self.delta)
        else:
            eps = 0

        print('Epsilon' + str(eps))
        return float(eps)

        # # Query for eps given delta
        # return PhiScalar(
        #     value=composed_mech.get_approxDP(self.delta),
        #     min_val=0,
        #     max_val=self.max_budget,
        #     entity=entity,
        # )

    def has_budget(self, entity_name: str, user_key: VerifyKey) -> bool:
        spend = self.get_eps_for_entity(entity_name, user_key=user_key)
        print("SPEND:" + str(spend))
        user_budget = self.entity2ledger.get_user_budget(user_key=user_key)
        print("BUDGET:" + str(user_budget))
        has_budget = spend < user_budget
        print("HAS Budget:" + str(has_budget))

        return has_budget

    def user_budget(self, user_key: VerifyKey):

        max_spend = 0

        for ent in self.entities:
            spend = self.get_eps_for_entity(entity_name=ent, user_key=user_key)
            if spend > max_spend:
                max_spend = spend

        return max_spend


    @property
    def entities(self) -> TypeKeysView[Entity]:
        return self.entity2ledger.keys()

    def overbudgeted_entities(self, user_key: VerifyKey) -> TypeSet[Entity]:
        entities = set()

        for entity_name in self.entities:
            if not self.has_budget(entity_name, user_key=user_key):
                entities.add(entity_name)

        return entities
    #
    # def print_ledger(self, delta: float = 1e-6) -> None:
    #     for entity, mechanisms in self.entity2ledger.items():
    #         print(str(entity) + "\t" + str(self.get_eps_for_entity(entity)))


class AccountantReference(RecursiveSerde):
    __attr_allowlist__ = ["msg"]

    def __init__(self, msg):
        self.msg = msg
