# stdlib
from collections.abc import KeysView
from typing import Any
from typing import List

# third party
from nacl.encoding import HexEncoder
from nacl.signing import VerifyKey
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

# relative
from syft.core.common.serde import _serialize
from syft.core.node.common.node_table.user import SyftUser

from ....adp.entity import Entity
from ..node_table.entity import Entity as EntitySchema
from ..node_table.ledger import Ledger
from ..node_table.mechanism import Mechanism as MechanismSchema

# from ..exceptions import SetupNotFoundError
from .database_manager import DatabaseManager


class EntityManager(DatabaseManager):
    schema = EntitySchema

    def __init__(self, database: Engine) -> None:
        super().__init__(db=database, schema=EntityManager.schema)

    def register(self, name: str):  # type: ignore
        entity = Entity(name=name)
        _obj = self._schema(name=entity.name)
        _obj.obj = entity
        session_local = sessionmaker(autocommit=False, autoflush=False, bind=self.db)()
        session_local.add(_obj)
        session_local.commit()
        session_local.close()
        return entity.name


class MechanismManager(DatabaseManager):
    schema = MechanismSchema

    def __init__(self, database: Engine) -> None:
        super().__init__(db=database, schema=MechanismManager.schema)

    def register(self, mechanism):  # type: ignore
        _obj = self._schema(entity_name=mechanism.entity, user_key=_serialize(mechanism.user_key, to_bytes=True))
        _obj.obj = mechanism
        session_local = sessionmaker(autocommit=False, autoflush=False, bind=self.db)()
        session_local.add(_obj)
        session_local.commit()
        obj_id = _obj.id
        session_local.close()
        return obj_id

    # def append(self, mech_id: int, mechanism: MechanismSchema) -> None:
    #     session_local = sessionmaker(autocommit=False, autoflush=False, bind=self.db)()
    #     mech_instance = session_local.query(self._schema).filter_by(id=mech_id).first()
    #     new_list = mech_instance.obj + [mechanism]
    #     mech_instance.obj = new_list
    #     session_local.flush()
    #     session_local.commit()
    #     session_local.close()


class LedgerManager(DatabaseManager):

    schema = Ledger

    def __init__(self, database: Engine) -> None:
        super().__init__(db=database, schema=LedgerManager.schema)
        self.entity_manager = EntityManager(database)
        self.mechanism_manager = MechanismManager(database)

    # def register(self, entity_name, mechanism_id) -> Any:
    #     _obj = self._schema(entity_name=entity_name, mechanism_id=mechanism_id)
    #     obj_id = _obj.id
    #     session_local = sessionmaker(autocommit=False, autoflush=False, bind=self.db)()
    #     session_local.add(_obj)
    #     session_local.commit()
    #     session_local.close()
    #     return obj_id

    def get_user_budget(self, user_key: VerifyKey):
        session_local = sessionmaker(autocommit=False, autoflush=False, bind=self.db)()
        serialized_key = user_key.encode(encoder=HexEncoder).decode("utf-8")
        budget = session_local.query(SyftUser).filter_by(verify_key=serialized_key).first().budget

        print("FOUND A USER WITH A BUDGET LIMITATION: " + str(budget))
        return budget

    # def append(self, entity_name, mechanism) -> None:
    #     ledger_instance = super().first(entity_name=entity_name)
    #     self.mechanism_manager.append(ledger_instance.mechanism_id, mechanism)

    def keys(self) -> KeysView:
        return KeysView(self.entity_manager.all())

    # def items(self) -> List:
    #     ledger_list = super().all()
    #     return [
    #         (ledger.entity_name, self.__getitem__(ledger.entity_name))
    #         for ledger in ledger_list
    #     ]

    def register_mechanisms(self, mechanisms:list):

        for m in mechanisms:

            # if entity doesn't eixst:
            if not self.entity_manager.contain(name=m.entity):

                # write to the entity table
                self.entity_manager.register(name=m.entity)

            self.mechanism_manager.register(m)

    def query(self, entity_name=None, user_key=None):

        if entity_name is not None and user_key is not None:
            return self.mechanism_manager.query(entity_name=entity_name, user_key=user_key)
        elif entity_name is not None and user_key is None:
            return self.mechanism_manager.query(entity_name=entity_name)
        elif entity_name is None and user_key is not None:
            return self.mechanism_manager.query(user_key=user_key)
        else:
            return self.mechanism_manager.all()


    # def __setitem__(self, entity_name, mechanisms: list):
    #     if super().contain(entity_name=entity_name):
    #         ledger_instance = super().first(entity_name=entity_name)
    #         self.entity_manager.delete(name=ledger_instance.entity_name)
    #         self.mechanism_manager.delete(id=ledger_instance.mechanism_id)
    #         super().delete(entity_name=entity_name)
    #
    #     for mechanism in mechanisms:
    #         entity_name = self.entity_manager.register(name=entity_name)
    #         mechanism_id = self.mechanism_manager.register(mechanism)
    #         self.register(entity_name=entity_name, mechanism_id=mechanism_id)
    #
    # def __getitem__(self, entity_name: str) -> Any:
    #     if super().contain(entity_name=entity_name):
    #         ledger_instance = super().first(entity_name=entity_name)
    #         return self.mechanism_manager.first(id=ledger_instance.mechanism_id).obj
    #     else:
    #         return None
