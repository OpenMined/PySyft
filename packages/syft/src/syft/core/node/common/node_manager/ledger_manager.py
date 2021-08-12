# stdlib
from typing import Any
from typing import List
from collections.abc import KeysView

# third party
from sqlalchemy import String
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

# relative
from ..node_table.ledger import Ledger
from ..node_table.entity import Entity as EntitySchema
from ..node_table.mechanism import Mechanism as MechanismSchema
from ....adp.entity import Entity

# from ..exceptions import SetupNotFoundError
from .database_manager import DatabaseManager

class EntityManager(DatabaseManager):
    schema = EntitySchema

    def __init__(self, database: Engine) -> None:
        super().__init__(db=database, schema=EntityManager.schema)

    def register(self, name) -> Any:
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

    def register(self, mechanism) -> Any:
        _obj = self._schema()
        _obj.obj = mechanism
        session_local = sessionmaker(autocommit=False, autoflush=False, bind=self.db)()
        session_local.add(_obj)
        session_local.commit()
        obj_id = _obj.id
        session_local.close()
        return obj_id
    
    def append(self, mech_id: int, mechanism) -> None:
        session_local = sessionmaker(autocommit=False, autoflush=False, bind=self.db)()
        mech_instance = session_local.query(self._schema).filter_by(id=mech_id).first()
        new_list = mech_instance.obj + [mechanism]
        mech_instance.obj = new_list
        session_local.flush()
        session_local.commit()
        session_local.close()
        
class LedgerManager(DatabaseManager):

    schema = Ledger

    def __init__(self, database: Engine) -> None:
        super().__init__(db=database, schema=LedgerManager.schema)
        self.entity_manager = EntityManager(database)
        self.mechanism_manager = MechanismManager(database)
    
    def register(self, entity_name, mechanism_id) -> Any:
        _obj = self._schema(
                entity_name=entity_name,
                mechanism_id=mechanism_id
        )
        obj_id = _obj.id
        session_local = sessionmaker(autocommit=False, autoflush=False, bind=self.db)()
        session_local.add(_obj)
        session_local.commit()
        session_local.close()
        return obj_id

    def append(self, entity_name, mechanism):
        ledger_instance = super().first(entity_name=entity_name)
        self.mechanism_manager.append(
                ledger_instance.mechanism_id,
                mechanism
        )

    def keys(self) -> KeysView:
        session_local = sessionmaker(autocommit=False, autoflush=False, bind=self.db)()
        keys = [entity[0] for entity in session_local.query(self._schema.entity_name).all()]
        return KeysView(keys)
    def items(self) -> List:
        ledger_list = super().all()
        return [ ( ledger.entity_name, self.__getitem__(ledger.entity_name) ) for ledger in ledger_list ]
        
    def __setitem__(self, entity_name, value):
        if super().contain(entity_name=entity_name):
            ledger_instance = super().first(entity_name=entity_name)
            self.entity_manager.delete(name=ledger_instance.entity_name)
            self.mechanism_manager.delete(id=ledger_instance.mechanism_id)
            super().delete(entity_name=entity_name)

        entity_name = self.entity_manager.register(name=entity_name)
        mechanism_id = self.mechanism_manager.register(value)
        result = self.register(
                entity_name=entity_name,
                mechanism_id=mechanism_id
        )

    def __getitem__(self, entity_name: str) -> Any:
        if super().contain(entity_name=entity_name):
            ledger_instance = super().first(entity_name=entity_name)
            return self.mechanism_manager.first(id=ledger_instance.mechanism_id).obj
        else:
            return None
