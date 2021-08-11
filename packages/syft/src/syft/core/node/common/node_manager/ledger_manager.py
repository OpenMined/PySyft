# stdlib
from typing import Any
from typing import List

# third party
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
        obj_id = _obj.id
        session_local = sessionmaker(autocommit=False, autoflush=False, bind=self.db)()
        session_local.add(_obj)
        session_local.commit()
        session_local.close()
        return obj_id

class LedgerManager(DatabaseManager):

    schema = Ledger

    def __init__(self, database: Engine) -> None:
        super().__init__(db=database, schema=LedgerManager.schema)
        self.entity_manager = EntityManager(database)
        self.mechanism_manager = MechanismManager(database)

    def __setitem__(self, entity_name, value):
        if super().contain(entity_name=entity_name):
            super().delete(entity_name=entity_name)

        entity_name = self.entity_manager.register(name=entity_name)
        mechanism_id = self.mechanism_manager.register(value)
        super().register(
                entity_name=entity_name,
                mechanism_id=mechanism_id
        )

    def __getitem__(self, key):
        if super().contain(entity_name=key):
            super().first(entity_name=key)
        else:
            return None
