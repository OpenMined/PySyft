# stdlib
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Type

# third party
from pymongo.collection import Collection
from pymongo.database import Database
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

# relative
from .....telemetry import instrument
from ..node_table import Base
from ..node_table.user import SyftObject


@instrument
class DatabaseManager:
    def __init__(self, schema: Type[Base], db: Engine) -> None:
        self._schema = schema
        self.db = db

    def register(self, **kwargs: Any) -> Any:
        """Register a new object into the database.

        Args:
            parameters : List of object parameters.
        Returns:
            object: Database Object
        """
        _obj = self._schema(**kwargs)
        session_local = sessionmaker(autocommit=False, autoflush=False, bind=self.db)()
        session_local.add(_obj)
        session_local.commit()
        return _obj

    def query(self, **kwargs: Any) -> List[Any]:
        """Query db objects filtering by parameters
        Args:
            parameters : List of parameters used to filter.
        """
        session_local = sessionmaker(autocommit=False, autoflush=False, bind=self.db)()
        objects = session_local.query(self._schema).filter_by(**kwargs).all()
        session_local.close()
        return objects

    def first(self, **kwargs: Any) -> Optional[Any]:
        """Query db objects filtering by parameters
        Args:
            parameters : List of parameters used to filter.
        """
        session_local = sessionmaker(autocommit=False, autoflush=False, bind=self.db)()
        objects = session_local.query(self._schema).filter_by(**kwargs).first()
        session_local.close()
        return objects

    def last(self, **kwargs: Any) -> Optional[Any]:
        """Query and return the last occurrence.

        Args:
            parameters: List of parameters used to filter.
        Return:
            obj: Last object instance.
        """
        session_local = sessionmaker(autocommit=False, autoflush=False, bind=self.db)()
        obj = session_local.query(self._schema).filter_by(**kwargs).all()[-1]
        session_local.close()
        return obj

    def all(self) -> List[Any]:
        session_local = sessionmaker(autocommit=False, autoflush=False, bind=self.db)()
        result = list(session_local.query(self._schema).all())
        session_local.close()
        return result

    def delete(self, **kwargs: Any) -> None:
        """Delete an object from the database.

        Args:
            parameters: Parameters used to filter the object.
        """
        session_local = sessionmaker(autocommit=False, autoflush=False, bind=self.db)()
        session_local.query(self._schema).filter_by(**kwargs).delete()
        session_local.commit()
        session_local.close()

    def modify(self, query: Dict[Any, Any], values: Dict[Any, Any]) -> None:
        """Modifies one or many records."""
        session_local = sessionmaker(autocommit=False, autoflush=False, bind=self.db)()
        session_local.query(self._schema).filter_by(**query).update(values)
        session_local.commit()
        session_local.close()

    def contain(self, **kwargs: Any) -> bool:
        session_local = sessionmaker(autocommit=False, autoflush=False, bind=self.db)()
        objects = session_local.query(self._schema).filter_by(**kwargs).all()
        session_local.close()
        return len(objects) != 0

    def __len__(self) -> int:
        session_local = sessionmaker(autocommit=False, autoflush=False, bind=self.db)()
        result = session_local.query(self._schema).count()
        session_local.close()
        return result

    def clear(self) -> None:
        local_session = sessionmaker(bind=self.db)()
        local_session.query(self._schema).delete()
        local_session.commit()
        local_session.close()


class NoSQLDatabaseManager:
    _collection_name: str
    _collection: Collection

    def __init__(self, db: Database) -> None:
        self._collection = db[self._collection_name]

    def add(self, obj: SyftObject) -> SyftObject:
        self._collection.insert_one(obj.to_mongo())
        return obj

    def drop(self) -> None:
        self._collection.drop()

    def delete(self, **search_params: Any) -> None:
        self._collection.delete_many(search_params)

    def update(self) -> None:
        pass

    def find(self, search_params: Dict[str, Any]) -> List[SyftObject]:
        results = []
        res = self._collection.find(search_params)
        for d in res:
            results.append(SyftObject.from_mongo(d))
        return results

    def find_one(self, search_params: Dict[str, Any]) -> Optional[SyftObject]:
        d = self._collection.find_one(search_params)
        if d is None:
            return d
        return SyftObject.from_mongo(d)

    def first(self, **search_params: Dict[str, Any]) -> Optional[SyftObject]:
        d = self._collection.find_one(search_params)
        if d is None:
            return d
        return SyftObject.from_mongo(d)

    def __len__(self) -> int:
        # empty document filter counts all documents.
        return self._collection.count_documents(filter={})

    def query(self, **search_params: Any) -> List[SyftObject]:
        """Query db objects filtering by parameters
        Args:
            parameters : List of parameters used to filter.
        """
        cursor = self._collection.find(search_params)
        objects = [SyftObject.from_mongo(obj) for obj in list(cursor)]
        return objects

    def update_one(self, query: dict, field_set: dict) -> None:
        self._collection.update_one(query, field_set)

    def all(self) -> List[SyftObject]:
        return self.find({})

    def clear(self) -> None:
        self.drop()

    def contain(self, **search_params: Any) -> bool:
        return bool(self.query(**search_params))
