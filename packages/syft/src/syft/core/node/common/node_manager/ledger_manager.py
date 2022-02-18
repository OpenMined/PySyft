# future
from __future__ import annotations

# stdlib
from collections.abc import KeysView
from typing import List
from typing import Optional

# third party
from nacl.encoding import HexEncoder
from nacl.signing import VerifyKey
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

# relative
from ....adp.entity import Entity
from ....adp.idp_gaussian_mechanism import iDPGaussianMechanism
from ....common.serde import _serialize
from ..node_table.entity import Entity as EntitySchema
from ..node_table.ledger import Ledger
from ..node_table.mechanism import Mechanism as MechanismSchema
from ..node_table.user import SyftUser
from .database_manager import DatabaseManager


class EntityManager(DatabaseManager):
    schema = EntitySchema

    def __init__(self, database: Engine) -> None:
        super().__init__(db=database, schema=EntityManager.schema)

    def register(self, name: str):  # type: ignore
        entity = Entity(name)  # Constructor
        _obj: EntitySchema = self._schema(name=entity.name)  # type: ignore
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


class AbstractLedger:
    def register_mechanisms(self, mechanisms: list) -> None:
        raise NotImplementedError

    def query(  # type: ignore
        self, entity_name: Optional[str] = None, user_key: Optional[str] = None
    ) -> List[iDPGaussianMechanism]:
        raise NotImplementedError

    def keys(self) -> KeysView:
        raise NotImplementedError

    def get_user_budget(self, user_key: VerifyKey) -> float:
        raise NotImplementedError


class DictLedger(AbstractLedger):
    def __init__(self) -> None:
        super().__init__()
        self._entity_dict: dict[str, iDPGaussianMechanism] = {}
        self._user_key_dict: dict[VerifyKey, iDPGaussianMechanism] = {}
        self._user_budget: dict[VerifyKey, float] = {}

    def register_mechanisms(self, mechanisms: list) -> None:
        for m in mechanisms:
            # if entity doesnt exist:
            if m.entity_name not in self._entity_dict:
                # write to the entity table
                self._entity_dict[m.entity_name] = m
            if m.user_key not in self._user_key_dict:
                self._user_key_dict[m.user_key] = m

    def query(
        self, entity_name: Optional[str] = None, user_key: Optional[VerifyKey] = None
    ) -> List[iDPGaussianMechanism]:
        mechanisms = []
        if entity_name is not None and user_key is None:
            if entity_name in self._entity_dict:
                mechanisms.append(self._entity_dict[entity_name])
                return mechanisms
        elif user_key is not None and entity_name is None:
            if user_key in self._user_key_dict:
                mechanisms.append(self._user_key_dict[user_key])
                return mechanisms
        elif entity_name is not None and user_key is not None:
            mechanisms.extend(self._entity_dict.values())
            m_set = set(mechanisms)
            return list(m_set.intersection(self._user_key_dict.values()))
        else:
            mechanisms.extend(self._entity_dict.values())
            mechanisms.extend(self._user_key_dict.values())
            return list(set(mechanisms))

        return mechanisms

    def keys(self) -> KeysView:
        return self._entity_dict.keys()

    def get_user_budget(self, user_key: VerifyKey) -> float:
        if user_key in self._user_budget:
            return self._user_budget[user_key]
        default_budget = 999999999
        print(f"Warning, no user found returning default budget. {default_budget}")
        return float(default_budget)


class DatabaseLedger(DatabaseManager, AbstractLedger):
    schema = Ledger

    def __init__(self, database: Engine) -> None:
        super().__init__(db=database, schema=DatabaseLedger.schema)
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
        return KeysView(self.entity_manager.all())  # type: ignore

    def register_mechanisms(self, mechanisms: list) -> None:
        for m in mechanisms:
            # if entity doesnt exist:
            if not self.entity_manager.contain(name=m.entity_name):
                # write to the entity table
                self.entity_manager.register(name=m.entity_name)
            self.mechanism_manager.register(m)  # type: ignore

    def query(  # type: ignore
        self, entity_name: Optional[str] = None, user_key: Optional[str] = None
    ) -> List[iDPGaussianMechanism]:
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
