# stdlib
from collections.abc import KeysView

# third party
from nacl.encoding import HexEncoder
from nacl.signing import VerifyKey
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

# syft absolute
from syft.core.common.serde import _serialize
from syft.core.node.common.node_table.user import SyftUser

# relative
from ....adp.entity import Entity
from ..node_table.entity import Entity as EntitySchema
from ..node_table.ledger import Ledger
from ..node_table.mechanism import Mechanism as MechanismSchema
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
        _obj = self._schema(
            entity_name=mechanism.entity_name,
            user_key=_serialize(mechanism.user_key, to_bytes=True),
        )
        _obj.obj = mechanism
        session_local = sessionmaker(autocommit=False, autoflush=False, bind=self.db)()
        session_local.add(_obj)
        session_local.commit()
        obj_id = _obj.id
        session_local.close()
        return obj_id


class LedgerManager(DatabaseManager):

    schema = Ledger

    def __init__(self, database: Engine) -> None:
        super().__init__(db=database, schema=LedgerManager.schema)
        self.entity_manager = EntityManager(database)
        self.mechanism_manager = MechanismManager(database)

    def get_user_budget(self, user_key: VerifyKey) -> float:
        session_local = sessionmaker(autocommit=False, autoflush=False, bind=self.db)()
        serialized_key = user_key.encode(encoder=HexEncoder).decode("utf-8")
        budget = (
            session_local.query(SyftUser)
            .filter_by(verify_key=serialized_key)
            .first()
            .budget
        )

        return budget

    def keys(self) -> KeysView:
        return KeysView(self.entity_manager.all())

    def register_mechanisms(self, mechanisms: list):

        for m in mechanisms:
            # if entity doesnt exist:
            if not self.entity_manager.contain(name=m.entity_name):
                # write to the entity table
                self.entity_manager.register(name=m.entity_name)
            self.mechanism_manager.register(m)

    def query(self, entity_name=None, user_key=None):
        if entity_name is not None and user_key is not None:
            return self.mechanism_manager.query(
                entity_name=entity_name, user_key=user_key
            )
        elif entity_name is not None and user_key is None:
            return self.mechanism_manager.query(entity_name=entity_name)
        elif entity_name is None and user_key is not None:
            return self.mechanism_manager.query(user_key=user_key)
        else:
            return self.mechanism_manager.all()
