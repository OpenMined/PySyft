# stdlib
from typing import Any
from typing import Collection
from typing import Iterable
from typing import List
from typing import Optional
from typing import cast

# third party
from pydantic import BaseSettings
from pymongo import MongoClient
import redis

# relative
from .....lib.python.dict import Dict as SyftDict
from ....common.serde.deserialize import _deserialize as deserialize
from ....common.serde.serialize import _serialize as serialize
from ....common.uid import UID
from ....store import ObjectStore
from ....store.proxy_dataset import ProxyDataset
from ....store.store_interface import StoreKey
from ....store.storeable_object import StorableObject
from .obj_metadata_manager import NoSQLObjectMetadataManager


class RedisStore(ObjectStore):
    def __init__(
        self,
        nosql_db_engine: MongoClient,
        db_name: str,
        settings: Optional[BaseSettings] = None,
    ) -> None:
        if settings is None:
            raise Exception("RedisStore requires Settings")
        self.settings = settings
        try:
            self.redis: redis.client.Redis = redis.Redis(
                host=settings.REDIS_HOST,
                port=self.settings.REDIS_PORT,
                db=self.settings.REDIS_STORE_DB_ID,
            )
            self.obj_metadata_manager = NoSQLObjectMetadataManager(
                nosql_db_engine, db_name
            )
        except Exception as e:
            print("failed to load redis", e)
            raise e

    def get_or_none(
        self, key: UID, proxy_only: bool = False
    ) -> Optional[StorableObject]:
        try:
            return self.get(key, proxy_only)
        except KeyError:
            return None

    def check_collision(self, key: UID) -> None:
        # Check ID collision with pointer's result.
        if self.get_or_none(key=key, proxy_only=True):
            raise Exception(
                "You're not allowed to perform this operation using this ID."
            )

    def get_objects_of_type(self, obj_type: type) -> Iterable[StorableObject]:
        # TODO: remove this kind of operation which pulls all the data out in one go
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
            all_values.append(self.get(key))

        return all_values

    def __contains__(self, key: UID) -> bool:
        return str(key.value) in self.redis

    def resolve_proxy_object(self, obj: Any) -> Any:
        obj = obj.get_s3_data(settings=self.settings)
        if obj is None:
            raise Exception(f"Failed to fetch real object from proxy. {type(obj)}")
        return obj

    def get(self, key: StoreKey, proxy_only: bool = False) -> StorableObject:
        key_str, key_uid = self.key_to_str_and_uid(key=key)

        obj = self.redis.get(key_str)
        obj_metadata = self.obj_metadata_manager.first(obj=key_str)

        if obj is None or obj_metadata is None:
            raise KeyError(f"Object not found! for UID: {key_str}")

        obj = deserialize(obj, from_bytes=True)
        if proxy_only is False and isinstance(obj, ProxyDataset):
            obj = self.resolve_proxy_object(obj=obj)

        obj = StorableObject(
            id=key_uid,
            data=obj,
            description=obj_metadata.description,
            tags=obj_metadata.tags,
            read_permissions=dict(
                deserialize(
                    bytes.fromhex(obj_metadata.read_permissions), from_bytes=True
                )
            ),
            search_permissions=dict(
                deserialize(
                    bytes.fromhex(obj_metadata.search_permissions), from_bytes=True
                )
            ),
            write_permissions=dict(
                deserialize(
                    bytes.fromhex(obj_metadata.write_permissions), from_bytes=True
                )
            ),
            # name=obj_metadata.name,
        )
        return obj

    def __getitem__(self, key: StoreKey) -> StorableObject:
        raise Exception("obj = store[key] not allowed because additional args required")

    # allow store[key] = obj
    # but not obj = store[key]
    def __setitem__(self, key: StoreKey, value: StorableObject) -> None:
        self.set(key=key, value=value)

    def set(self, key: StoreKey, value: StorableObject) -> None:
        key_str, _ = self.key_to_str_and_uid(key=key)

        is_proxy_dataset = False
        if isinstance(value._data, ProxyDataset):
            bin = serialize(value._data, to_bytes=True)
            is_proxy_dataset = True
        else:
            bin = serialize(value.data, to_bytes=True)
        self.redis.set(key_str, bin)  # type: ignore

        if not self.obj_metadata_manager.contain(obj=key_str):
            read_permissions = cast(
                bytes,
                serialize(SyftDict(value.read_permissions), to_bytes=True),
            ).hex()
            search_permissions = cast(
                bytes,
                serialize(SyftDict(value.search_permissions), to_bytes=True),
            ).hex()
            write_permissions = cast(
                bytes,
                serialize(SyftDict(value.write_permissions), to_bytes=True),
            ).hex()

            self.obj_metadata_manager.create_metadata(
                obj=key_str,
                tags=value.tags,
                description=value.description,
                read_permissions=read_permissions,
                search_permissions=search_permissions,
                write_permissions=write_permissions,
                is_proxy_dataset=is_proxy_dataset,
            )

    def delete(self, key: UID) -> None:
        try:
            # Yellow Team: fix this, update to use: obj_metadata_manager?
            # self.dataset_manager.delete_bin_obj(bin_obj_id=key)
            # Check if the uploaded data is a proxy dataset
            # if metadata_to_delete and metadata_to_delete.is_proxy_dataset:
            #     # Retrieve proxy dataset from store
            #     obj = self.get(key=key, proxy_only=True)
            #     proxy_dataset = obj.data
            #     if proxy_dataset:
            #         proxy_dataset.delete_s3_data(settings=self.settings)

            self.redis.delete(str(key.value))
        except Exception as e:
            print(f"{type(self)} Exception in __delitem__ error {key}. {e}")

    def clear(self) -> None:
        self.redis.flushdb()
        self.obj_metadata_manager.clear()

    def __repr__(self) -> str:
        return f"{type(self)}"
