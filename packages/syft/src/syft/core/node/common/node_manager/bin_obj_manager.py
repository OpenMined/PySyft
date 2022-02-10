# stdlib
from typing import Any
from typing import Collection
from typing import Iterable
from typing import List
from typing import Optional
from typing import Union
from typing import cast

# third party
import redis
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm.session import Session
from torch import Tensor

# syft absolute
import syft

# relative
from ....common.uid import UID
from ....node.common.node_table.bin_obj_dataset import BinObjDataset
from ....store import ObjectStore
from ....store.storeable_object import StorableObject
from ..node_table.bin_obj_metadata import ObjectMetadata

ENCODING = "UTF-8"


def create_storable(
    _id: UID, data: Tensor, description: str, tags: Optional[List[str]] = None
) -> StorableObject:
    obj = StorableObject(id=_id, data=data, description=description, tags=tags)

    return obj


class DictStore(ObjectStore):
    def __init__(self, db: Session) -> None:
        self.db = db
        self.kv_store: dict[UID, Any] = {}

    def get_object(self, key: UID) -> Optional[StorableObject]:
        try:
            return self.__getitem__(key)
        except KeyError as e:  # noqa: F841
            return None

    def get_objects_of_type(self, obj_type: type) -> Iterable[StorableObject]:
        # raise NotImplementedError("get_objects_of_type")
        # return [obj for obj in self.values() if isinstance(obj.data, obj_type)]
        return self.values()

    def __sizeof__(self) -> int:
        return self.values().__sizeof__()

    def __str__(self) -> str:
        return f"{type(self)}"

    def __len__(self) -> int:
        return len(self.kv_store.keys())

    def keys(self) -> Collection[UID]:
        return self.kv_store.keys()

    def values(self) -> List[StorableObject]:
        keys = self.kv_store.keys()
        # this is bad we need to decouple getting the data from the search
        all_values = []
        for key in keys:
            all_values.append(self.__getitem__(key))

        return all_values

    def __contains__(self, key: UID) -> bool:
        return key in self.kv_store

    def __getitem__(self, key: Union[UID, str, bytes]) -> StorableObject:
        try:
            local_session = sessionmaker(bind=self.db)()
            # bin_obj = local_session.query(BinObject).filter_by(id=str(key.value)).first()

            if isinstance(key, UID):
                key_str = str(key.value)
                key_uid = key
            elif isinstance(key, bytes):
                key_str = str(key.decode("utf-8"))
                key_uid = UID.from_string(key_str)
            else:
                key_str = key
                key_uid = UID.from_string(key_str)

            obj = self.kv_store[key_uid]
            obj_metadata = (
                local_session.query(ObjectMetadata).filter_by(obj=key_str).first()
            )

            if obj is None or obj_metadata is None:
                raise KeyError(f"Object not found! for UID: {key_uid}")

            obj = StorableObject(
                id=key_uid,
                data=obj.data,
                description=obj_metadata.description,
                tags=obj_metadata.tags,
                read_permissions=dict(
                    syft.deserialize(
                        bytes.fromhex(obj_metadata.read_permissions), from_bytes=True
                    )
                ),
                search_permissions=dict(
                    syft.deserialize(
                        bytes.fromhex(obj_metadata.search_permissions), from_bytes=True
                    )
                ),
                # name=obj_metadata.name,
            )
            local_session.close()
            return obj
        except Exception as e:
            print(f"Cant store object {str(key)}", e)
            raise KeyError(f"Object not found! for UID: {str(key)}")

    def is_dataset(self, key: UID) -> bool:
        local_session = sessionmaker(bind=self.db)()
        is_dataset_obj = (
            local_session.query(BinObjDataset).filter_by(obj=str(key.value)).exists()
        )
        is_dataset_obj = local_session.query(is_dataset_obj).scalar()
        local_session.close()
        return is_dataset_obj

    def _get_obj_dataset_relation(self, key: UID) -> Optional[BinObjDataset]:
        local_session = sessionmaker(bind=self.db)()
        obj_dataset_relation = (
            local_session.query(BinObjDataset).filter_by(obj=str(key.value)).first()
        )
        local_session.close()
        return obj_dataset_relation

    def __setitem__(self, key: UID, value: StorableObject) -> None:
        try:
            self.kv_store[key] = value
        except Exception as e:
            print(f"failed to add {key} and value to dict", e)

        metadata_obj = ObjectMetadata(
            obj=str(key.value),
            tags=value.tags,
            description=value.description,
            read_permissions=cast(
                bytes,
                syft.serialize(
                    syft.lib.python.Dict(value.read_permissions), to_bytes=True
                ),
            ).hex(),
            search_permissions=cast(
                bytes,
                syft.serialize(
                    syft.lib.python.Dict(value.search_permissions), to_bytes=True
                ),
            ).hex(),
            # name=metadata_dict["name"],
        )

        obj_dataset_relation = self._get_obj_dataset_relation(key)
        if obj_dataset_relation:
            # Create a object dataset relationship for the new object
            obj_dataset_relation = BinObjDataset(
                id=obj_dataset_relation.id,
                name=obj_dataset_relation.name,
                obj=str(key.value),
                dataset=obj_dataset_relation.dataset,
                dtype=obj_dataset_relation.dtype,
                shape=obj_dataset_relation.shape,
            )

        local_session = sessionmaker(bind=self.db)()
        local_session.add(metadata_obj)
        local_session.add(obj_dataset_relation) if obj_dataset_relation else None
        local_session.commit()
        local_session.close()

    def delete(self, key: UID) -> None:
        try:
            del self.kv_store[key]
            local_session = sessionmaker(bind=self.db)()
            metadata_to_delete = (
                local_session.query(ObjectMetadata)
                .filter_by(obj=str(key.value))
                .first()
            )
            local_session.delete(metadata_to_delete)
            local_session.commit()
            local_session.close()
        except Exception as e:
            print(f"{type(self)} Exception in __delitem__ error {key}. {e}")

    def clear(self) -> None:
        self.kv_store = {}
        local_session = sessionmaker(bind=self.db)()
        local_session.query(ObjectMetadata).delete()
        local_session.commit()
        local_session.close()

    def __repr__(self) -> str:
        return f"{type(self)}"


class RedisStore(ObjectStore):
    def __init__(self, db: Session) -> None:
        self.db = db
        print("connecting to redis")
        try:
            self.redis: redis.client.Redis = redis.Redis(host="redis", port=6379)
        except Exception as e:
            print("failed to load redis", e)
            raise e

    def get_object(self, key: UID) -> Optional[StorableObject]:
        try:
            return self.__getitem__(key)
        except KeyError as e:  # noqa: F841
            return None

    def get_objects_of_type(self, obj_type: type) -> Iterable[StorableObject]:
        # raise NotImplementedError("get_objects_of_type")
        # return [obj for obj in self.values() if isinstance(obj.data, obj_type)]
        return self.values()

    def __sizeof__(self) -> int:
        return self.values().__sizeof__()

    def __str__(self) -> str:
        return f"{type(self)}"

    def __len__(self) -> int:
        return self.redis.dbsize()

    def keys(self) -> Collection[UID]:
        key_bytes = self.redis.keys()
        key_ids = [UID.from_string(str(key.decode("utf-8"))) for key in key_bytes]
        return key_ids

    def values(self) -> List[StorableObject]:
        key_bytes = self.redis.keys()
        # this is bad we need to decouple getting the data from the search
        all_values = []
        for key in key_bytes:
            all_values.append(self.__getitem__(key))

        return all_values

    def __contains__(self, key: UID) -> bool:
        return str(key.value) in self.redis

    def __getitem__(self, key: Union[UID, str, bytes]) -> StorableObject:
        local_session = sessionmaker(bind=self.db)()

        if isinstance(key, UID):
            key_str = str(key.value)
            key_uid = key
        elif isinstance(key, bytes):
            key_str = str(key.decode("utf-8"))
            key_uid = UID.from_string(key_str)
        else:
            key_str = key
            key_uid = UID.from_string(key_str)

        obj = self.redis.get(key_str)
        obj_metadata = (
            local_session.query(ObjectMetadata).filter_by(obj=key_str).first()
        )
        if obj is None or obj_metadata is None:
            raise KeyError(f"Object not found! for UID: {key_str}")

        obj = syft.deserialize(obj, from_bytes=True)

        obj = StorableObject(
            id=key_uid,
            data=obj,
            description=obj_metadata.description,
            tags=obj_metadata.tags,
            read_permissions=dict(
                syft.deserialize(
                    bytes.fromhex(obj_metadata.read_permissions), from_bytes=True
                )
            ),
            search_permissions=dict(
                syft.deserialize(
                    bytes.fromhex(obj_metadata.search_permissions), from_bytes=True
                )
            ),
            # name=obj_metadata.name,
        )
        local_session.close()
        return obj

    def is_dataset(self, key: UID) -> bool:
        local_session = sessionmaker(bind=self.db)()
        is_dataset_obj = (
            local_session.query(BinObjDataset).filter_by(obj=str(key.value)).exists()
        )
        is_dataset_obj = local_session.query(is_dataset_obj).scalar()
        local_session.close()
        return is_dataset_obj

    def _get_obj_dataset_relation(self, key: UID) -> Optional[BinObjDataset]:
        local_session = sessionmaker(bind=self.db)()
        obj_dataset_relation = (
            local_session.query(BinObjDataset).filter_by(obj=str(key.value)).first()
        )
        local_session.close()
        return obj_dataset_relation

    def __setitem__(self, key: UID, value: StorableObject) -> None:

        bin = syft.serialize(value.data, to_bytes=True)
        self.redis.set(str(key.value), bin)

        metadata_obj = ObjectMetadata(
            obj=str(key.value),
            tags=value.tags,
            description=value.description,
            read_permissions=cast(
                bytes,
                syft.serialize(
                    syft.lib.python.Dict(value.read_permissions), to_bytes=True
                ),
            ).hex(),
            search_permissions=cast(
                bytes,
                syft.serialize(
                    syft.lib.python.Dict(value.search_permissions), to_bytes=True
                ),
            ).hex(),
            # name=metadata_dict["name"],
        )

        obj_dataset_relation = self._get_obj_dataset_relation(key)
        if obj_dataset_relation:
            # Create a object dataset relationship for the new object
            obj_dataset_relation = BinObjDataset(
                # id=obj_dataset_relation.id,  NOTE: Commented temporarily
                name=obj_dataset_relation.name,
                obj=str(key.value),
                dataset=obj_dataset_relation.dataset,
                dtype=obj_dataset_relation.dtype,
                shape=obj_dataset_relation.shape,
            )

        local_session = sessionmaker(bind=self.db)()
        local_session.add(metadata_obj)
        local_session.add(obj_dataset_relation) if obj_dataset_relation else None
        local_session.commit()
        local_session.close()

    def delete(self, key: UID) -> None:
        try:
            self.redis.delete(str(key.value))
            local_session = sessionmaker(bind=self.db)()
            metadata_to_delete = (
                local_session.query(ObjectMetadata)
                .filter_by(obj=str(key.value))
                .first()
            )
            local_session.delete(metadata_to_delete)
            local_session.commit()
            local_session.close()
        except Exception as e:
            print(f"{type(self)} Exception in __delitem__ error {key}. {e}")

    def clear(self) -> None:
        self.redis.flushdb()
        local_session = sessionmaker(bind=self.db)()
        local_session.query(ObjectMetadata).delete()
        local_session.commit()
        local_session.close()

    def __repr__(self) -> str:
        return f"{type(self)}"
