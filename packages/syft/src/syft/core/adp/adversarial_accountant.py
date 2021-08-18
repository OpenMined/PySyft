# CLEANUP NOTES:
# - remove unused comments
# - add documentation for each method
# - add comments inline explaining each piece
# - add a unit test for each method (at least)

# stdlib
from typing import Dict as TypeDict
from typing import KeysView as TypeKeysView
from typing import List as TypeList
from typing import Optional
from typing import Set as TypeSet

# third party
from autodp.autodp_core import Mechanism
from autodp.transformer_zoo import Composition
from nacl.signing import VerifyKey
from sqlalchemy.engine import Engine

# relative
from ..common.serde.recursive import RecursiveSerde
from ..node.common.node_manager.ledger_manager import LedgerManager
from .entity import Entity
from .idp_gaussian_mechanism import iDPGaussianMechanism
from .scalar import PhiScalar


class AdversarialAccountant:
    def __init__(
        self, db_engine: Engine, max_budget: float = 10, delta: float = 1e-6
    ) -> None:

        # this is a database-backed lookup table
        self.entity2ledger = LedgerManager(db_engine)

        # this is a temporary lookup table for mechanisms we're not sure
        # we're going to keep (See publish.py for how this is used)
        self.temp_entity2ledger = {}
        self.max_budget = max_budget
        self.delta = delta

    def temp_append(
        self, entity2mechanisms: TypeDict[Entity, TypeList[Mechanism]]
    ) -> None:
        for key, ms in entity2mechanisms.items():
            if key not in self.temp_entity2ledger.keys():
                self.temp_entity2ledger[key] = list()
            for m in ms:
                self.temp_entity2ledger[key].append(m)

    def append(self, entity2mechanisms: TypeDict[str, TypeList[Mechanism]]) -> None:
        mechanisms = list()
        for _, ms in entity2mechanisms.items():
            for m in ms:
                mechanisms.append(m)

        self.entity2ledger.register_mechanisms(mechanisms)

    def save_temp_ledger_to_longterm_ledger(self):
        self.append(entity2mechanisms=self.temp_entity2ledger)

    def get_eps_for_entity(
        self, entity: Entity, user_key: Optional[VerifyKey] = None
    ) -> PhiScalar:
        # compose them with the transformation: compose
        compose = Composition()

        # fetch mechanisms from the database
        table_mechanisms = self.entity2ledger.query(entity_name=entity.name)
        mechanisms = [x.obj for x in table_mechanisms]

        # filter out mechanisms that weren't created by this data scientist user
        if user_key is not None:
            filtered_mechanisms = []
            for mech in mechanisms:
                if mech.user_key == user_key:
                    filtered_mechanisms.append(mech)

            mechanisms = filtered_mechanisms

        # print("Num mechanisms before TEMP:" + str(len(mechanisms)))
        # print("self.temp_entity2ledger:" + str(self.temp_entity2ledger))

        if entity in self.temp_entity2ledger.keys():
            mechanisms = mechanisms + self.temp_entity2ledger[entity]

        # # filter out mechanisms that weren't created by this data scientist user
        # if user_key is not None:
        #     filtered_mechanisms = []
        #     for mech in mechanisms:
        #         # left = _deserialize(mech.user_key, from_bytes=True)
        #         # print("Comparing Left:" + str(mech.user_key) + " of type " + str(type(mech.user_key)))
        #         # print("Comparing Right:" + str(user_key) + " of type " + str(type(user_key)))

        #         # user_key = VerifyKey(user_key.encode("utf-8"), encoder=HexEncoder)

        #         if mech.user_key == user_key:
        #             filtered_mechanisms.append(mech)

        #     mechanisms = filtered_mechanisms

        # print("Num mechanisms after TEMP:" + str(len(mechanisms)))
        # for m in mechanisms:
        # print("Filtered Mechanism Entity:" + str(m.entity_name))

        # print("Mechanisma after filtering: ", mechanisms)
        # use verify key to specify the user
        # for all entities in the db,
        # how do we ensure that no data scientist
        # exceeds the budget of any entity?

        # map dataset
        if len(mechanisms) > 0:
            composed_mech = compose(mechanisms, [1] * len(mechanisms))
            eps = composed_mech.get_approxDP(self.delta)

            # if we have a user key, get the budget and clamp the epsilon
            # if user_key is not None:
            #     user_budget = self.entity2ledger.get_user_budget(user_key=user_key)
            #     print("USER BUDGET:" + str(user_budget))
            #     print(f"EPS: {eps}")
            #     if eps > user_budget:
            #         print("setting the eps to the user budget")
            #         eps = user_budget
        else:
            eps = 0

        # print("Epsilon" + str(eps))
        return float(eps)

        # # Query for eps given delta
        # return PhiScalar(
        #     value=composed_mech.get_approxDP(self.delta),
        #     min_val=0,
        #     max_val=self.max_budget,
        #     entity=entity,
        # )

    def has_budget(self, entity: Entity, user_key: VerifyKey) -> bool:
        spend = self.get_eps_for_entity(entity=entity, user_key=user_key)
        # print("SPEND:" + str(spend))
        user_budget = self.entity2ledger.get_user_budget(user_key=user_key)
        # print("USER BUDGET:" + str(user_budget))
        # print("ACCOUNTANT MAX BUDGET", self.max_budget)
        # @Andrew can we use <= or does it have to be <
        has_budget = spend <= user_budget
        # print(f"has_budget = {spend} < {user_budget}")

        return has_budget

    def user_budget(self, user_key: VerifyKey):
        max_spend = 0

        for ent in self.entities:
            spend = self.get_eps_for_entity(entity=ent, user_key=user_key)
            if spend > max_spend:
                max_spend = spend

        return max_spend

    @property
    def entities(self) -> TypeKeysView[Entity]:
        return self.entity2ledger.keys()

    def overbudgeted_entities(
        self,
        temp_entities: TypeDict[Entity, TypeList[iDPGaussianMechanism]],
        user_key: VerifyKey,
    ) -> TypeSet[Entity]:
        entities = set()

        for entity, _ in temp_entities.items():
            if not self.has_budget(entity, user_key=user_key):
                entities.add(entity)

        return entities

    def print_ledger(self, delta: float = 1e-6) -> None:
        for mechanism in self.entity2ledger.mechanism_manager.all():
            entity = self.entity2ledger.entity_manager.first(name=mechanism.entity_name)
            print(
                str(mechanism.entity_name) + "\t" + str(self.get_eps_for_entity(entity))
            )


class AccountantReference(RecursiveSerde):
    __attr_allowlist__ = ["msg"]

    def __init__(self, msg) -> None:
        self.msg = msg
