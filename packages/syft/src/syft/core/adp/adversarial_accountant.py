# CLEANUP NOTES:
# - remove unused comments
# - add documentation for each method
# - add comments inline explaining each piece
# - add a unit test for each method (at least)

# stdlib
from functools import lru_cache
import math
from typing import Dict as TypeDict
from typing import Iterable
from typing import KeysView as TypeKeysView
from typing import List as TypeList
from typing import Optional
from typing import Set as TypeSet
from typing import Tuple
from typing import Union

# third party
from autodp.autodp_core import Mechanism
from autodp.transformer_zoo import Composition
from nacl.signing import VerifyKey
from sqlalchemy.engine import Engine

# relative
from ..node.common.node_manager.ledger_manager import AbstractLedger
from ..node.common.node_manager.ledger_manager import DatabaseLedger
from ..node.common.node_manager.ledger_manager import DictLedger
from .entity import DataSubjectGroup
from .entity import Entity
from .idp_gaussian_mechanism import iDPGaussianMechanism


def compose_mechanisms(
    mechanisms: Iterable[iDPGaussianMechanism], delta: float
) -> float:
    sigmas = list()
    squared_l2_norms = list()
    squared_l2_norm_upper_bounds = list()
    Ls = list()
    values = list()

    for m in mechanisms:
        sigmas.append(m.params["sigma"])
        squared_l2_norms.append(m.params["private_value"])
        squared_l2_norm_upper_bounds.append(m.params["public_value"])
        Ls.append(m.params["L"])
        values.append(m.params["value"])

    return compose_mechanisms_via_simplified_args_for_lru_cache(
        tuple(sigmas),
        tuple(squared_l2_norms),
        tuple(squared_l2_norm_upper_bounds),
        tuple(Ls),
        tuple(values),
        delta,
    )


@lru_cache(maxsize=None)
def compose_mechanisms_via_simplified_args_for_lru_cache(
    sigmas: Tuple[float],
    squared_l2_norms: Tuple[float],
    squared_l2_norm_upper_bounds: Tuple[float],
    Ls: Tuple[float],
    values: Tuple[float],
    delta: float,
) -> float:
    mechanisms = list()
    for i in range(len(sigmas)):

        m = iDPGaussianMechanism(
            sigma=sigmas[i],
            squared_l2_norm=squared_l2_norms[i],
            squared_l2_norm_upper_bound=squared_l2_norm_upper_bounds[i],
            L=Ls[i],
            entity_name="",
        )
        m.params["value"] = values[i]
        mechanisms.append(m)
    # compose them with the transformation: compose
    compose = Composition()
    composed_mech = compose(mechanisms, [1] * len(mechanisms))
    eps = composed_mech.get_approxDP(delta)
    return eps


class AdversarialAccountant:
    """Adversarial Accountant class that keeps track of budget and maintains a privacy ledger."""

    def __init__(
        self, db_engine: Engine = None, max_budget: float = 10, delta: float = 1e-6
    ) -> None:

        if db_engine is not None:
            # this is a database-backed lookup table
            # maps an entity to an actual budget
            self.entity2ledger: AbstractLedger = DatabaseLedger(db_engine)
        else:
            self.entity2ledger: AbstractLedger = DictLedger()  # type: ignore

        # this is a temporary lookup table for mechanisms we're not sure
        # we're going to keep (See publish.py for how this is used)
        self.temp_entity2ledger: TypeDict = {}
        self.max_budget = max_budget
        self.delta = delta

    def temp_append(
        self, entity2mechanisms: TypeDict[Entity, TypeList[Mechanism]]
    ) -> None:
        # add the mechanisms to all of the entities
        for key, ms in entity2mechanisms.items():
            if key not in self.temp_entity2ledger.keys():
                self.temp_entity2ledger[key] = list()
            for m in ms:
                self.temp_entity2ledger[key].append(m)

    def append(self, entity2mechanisms: TypeDict[str, TypeList[Mechanism]]) -> None:
        mechanisms = list()
        # add all the mechanisms
        for _, ms in entity2mechanisms.items():
            for m in ms:
                mechanisms.append(m)

        self.entity2ledger.register_mechanisms(mechanisms)

    # save the temporary ledger into the database
    def save_temp_ledger_to_longterm_ledger(self) -> None:
        self.append(entity2mechanisms=self.temp_entity2ledger)

    # return epsilons for each entity
    def get_eps_for_entity(
        self,
        entity: Entity,
        user_key: Optional[VerifyKey] = None,
        returned_epsilon_is_private: bool = False,
    ) -> float:

        # fetch mechanisms from the database
        table_mechanisms = self.entity2ledger.query(entity_name=entity.name)  # type: ignore
        mechanisms = [x.obj for x in table_mechanisms]

        # filter out mechanisms that weren't created by this data scientist user
        if user_key is not None:
            filtered_mechanisms = []
            for mech in mechanisms:
                if mech.user_key == user_key:
                    filtered_mechanisms.append(mech)

            mechanisms = filtered_mechanisms

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

        if returned_epsilon_is_private:
            for mech in mechanisms:
                mech.params["value"] = mech.params["private_value"]
        else:
            for mech in mechanisms:
                mech.params["value"] = mech.params["public_value"]

        # map dataset
        if len(mechanisms) > 0:
            eps = compose_mechanisms(tuple(mechanisms), self.delta)

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

        return float(eps)

        # # Query for eps given delta
        # return PhiScalar(
        #     value=composed_mech.get_approxDP(self.delta),
        #     min_val=0,
        #     max_val=self.max_budget,
        #     entity=entity,
        # )

    # checks if the entity has budget or not
    def has_budget(
        self,
        entity: Entity,
        user_key: VerifyKey,
        returned_epsilon_is_private: bool = False,
    ) -> bool:

        spent = self.get_eps_for_entity(
            entity=entity,
            user_key=user_key,
            returned_epsilon_is_private=returned_epsilon_is_private,
        )

        user_budget = self.entity2ledger.get_user_budget(user_key=user_key)

        # print("ACCOUNTANT MAX BUDGET", self.max_budget)
        # @Andrew can we use <= or does it have to be <
        has_budget = spent <= user_budget
        # print("\n\nHas Budget:" + str(has_budget))
        # print("YOU'VE SPENT:" + str(spent))
        # print("USER BUDGET:" + str(user_budget))
        return has_budget

    # returns maximum entity epsilon
    def user_budget(
        self, user_key: VerifyKey, returned_epsilon_is_private: bool = False
    ) -> float:
        max_spend = 0.0

        for ent in self.entities:
            spend = self.get_eps_for_entity(
                entity=ent,
                user_key=user_key,
                returned_epsilon_is_private=returned_epsilon_is_private,
            )
            if math.isnan(spend) or math.isinf(spend):
                print(f"Warning: Spend is {spend}")

            if spend > max_spend:
                max_spend = float(spend)

        return float(max_spend)

    def get_remaining_budget(
        self, user_key: VerifyKey, returned_epsilon_is_private: bool = False
    ) -> float:
        max_spend = self.user_budget(
            user_key=user_key, returned_epsilon_is_private=returned_epsilon_is_private
        )
        if math.isnan(max_spend) or math.isinf(max_spend):
            print(f"Warning: Remaining budget not valid with max_spend {max_spend}")

        # If there's no entity registered
        if not len(self.entities):
            max_spend = 0

        user_budget = self.entity2ledger.get_user_budget(user_key=user_key)

        return float(user_budget - max_spend)

    @property
    def entities(self) -> TypeKeysView[Entity]:
        return self.entity2ledger.keys()

    # returns a collection of entities having no budget
    def overbudgeted_entities(
        self,
        temp_entities: Union[
            TypeDict[Entity, TypeList[iDPGaussianMechanism]],
            TypeDict[DataSubjectGroup, TypeList[iDPGaussianMechanism]],
        ],
        user_key: VerifyKey,
        returned_epsilon_is_private: bool = False,
    ) -> Union[TypeSet[Entity], TypeSet[DataSubjectGroup]]:
        entities = set()

        for entity, _ in temp_entities.items():
            if isinstance(entity, DataSubjectGroup):
                for e in entity.entity_set:
                    if not self.has_budget(
                        e,
                        user_key=user_key,
                        returned_epsilon_is_private=returned_epsilon_is_private,
                    ):
                        entities.add(
                            entity
                        )  # Leave out the whole group if ANY of its entities are over budget
            elif isinstance(entity, Entity):
                if not self.has_budget(
                    entity,
                    user_key=user_key,
                    returned_epsilon_is_private=returned_epsilon_is_private,
                ):
                    entities.add(entity)  # type: ignore
            else:
                raise Exception
        return entities

    # prints entity and its epsilon value
    def print_ledger(self, returned_epsilon_is_private: bool = False) -> None:
        for entity in self.entities:
            if entity is not None:
                print(
                    str(entity.name)
                    + "\t"
                    + str(
                        self.get_eps_for_entity(
                            entity=entity,
                            returned_epsilon_is_private=returned_epsilon_is_private,
                        )
                    )
                )
