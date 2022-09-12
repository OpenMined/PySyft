# stdlib
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Type
from typing import Union
from uuid import UUID

# third party
from pymongo.collection import Collection
from pymongo.mongo_client import MongoClient
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

# relative
from .....telemetry import instrument
from ....common.uid import UID
from ..exceptions import ObjectNotFoundError
from ..node_table import Base
from ..node_table.syft_object import SyftObject


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


def convert_to_uuid(value: Union[str, UID, UUID]) -> UUID:
    if isinstance(value, str):
        return UUID(value)
    if isinstance(value, UID):
        return value.value
    elif isinstance(value, UUID):
        return value
    else:
        # Ask @Madhava , can we check for  invalid types , even though type annotation is specified.
        return ValueError(  # type: ignore
            f"Invalid value,type:{value,type(value)} for Conversion UUID expected Union[str,UID,UUID]"
        )


def convert_to_mongo_id(fields: Dict[Any, Any]) -> Dict:
    if "id" in fields:
        fields["_id"] = convert_to_uuid(fields["id"])
        del fields["id"]
    return fields


class NoSQLDatabaseManager:
    _collection_name: str
    _collection: Collection

    def __init__(self, client: MongoClient, db_name: str) -> None:
        self._client = client
        self._database = client[db_name]
        self._collection = self._database[self._collection_name]

    def add(self, obj: SyftObject) -> SyftObject:
        self._collection.insert_one(obj.to_mongo())
        return obj

    def drop(self) -> None:
        self._collection.drop()

    def delete(self, **search_params: Any) -> None:
        search_params = convert_to_mongo_id(search_params)
        self._collection.delete_many(search_params)

    def update(
        self, search_params: Dict[str, Any], updated_args: Dict[str, Any]
    ) -> None:
        search_params = convert_to_mongo_id(search_params)
        obj = self.first(**search_params)
        attributes: Dict[str, Any] = {}
        for k, v in updated_args.items():
            if k not in obj.__attr_state__:  # type: ignore
                raise ValueError(f"Cannot set an non existing field:{k} to Syft Object")
            else:
                setattr(obj, k, v)
            if k in obj.__attr_searchable__:  # type: ignore
                attributes[k] = v
        attributes["__blob__"] = obj.to_bytes()  # type: ignore
        attributes = convert_to_mongo_id(attributes)
        self.update_one(query=search_params, values=attributes)

    def find(self, search_params: Dict[str, Any]) -> List[SyftObject]:
        search_params = convert_to_mongo_id(search_params)
        results = []
        res = self._collection.find(search_params)
        for d in res:
            results.append(SyftObject.from_mongo(d))
        return results

    def find_one(self, search_params: Dict[str, Any]) -> Optional[SyftObject]:
        search_params = convert_to_mongo_id(search_params)
        d = self._collection.find_one(search_params)
        if d is None:
            return d
        return SyftObject.from_mongo(d)

    def first(self, **search_params: Any) -> Optional[SyftObject]:
        search_params = convert_to_mongo_id(search_params)
        d = self._collection.find_one(search_params)
        if d is None:
            raise ObjectNotFoundError
        return SyftObject.from_mongo(d)

    def __len__(self) -> int:
        # empty document filter counts all documents.
        return self._collection.count_documents(filter={})

    def query(self, **search_params: Any) -> List[SyftObject]:
        """Query db objects filtering by parameters
        Args:
            parameters : List of parameters used to filter.
        """
        search_params = convert_to_mongo_id(search_params)
        cursor = self._collection.find(search_params)
        objects = [SyftObject.from_mongo(obj) for obj in list(cursor)]
        return objects

    def update_one(self, query: dict, values: dict) -> None:
        query = convert_to_mongo_id(query)
        values = convert_to_mongo_id(values)
        values = {"$set": values}
        self._collection.update_one(query, values)

    def all(self) -> List[SyftObject]:
        return self.find({})

    def clear(self) -> None:
        self.drop()

    def contain(self, **search_params: Any) -> bool:
        return bool(self.query(**search_params))
