# stdlib
from copy import deepcopy
from typing import Any
from typing import Collection
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional

# third party
from pydantic import BaseSettings
from pymongo import MongoClient

# relative
from ....common.serde.deserialize import _deserialize as deserialize
from ....common.serde.serialize import _serialize as serialize
from ....common.uid import UID
from ....store import ObjectStore
from ....store.proxy_dataset import ProxyDataset
from ....store.store_interface import StoreKey
from ....store.storeable_object import StorableObject


class DictStore(ObjectStore):
    def __init__(
        self,
        settings: BaseSettings,
        nosql_db_engine: MongoClient,
        db_name: str,
    ) -> None:
        self.settings = settings
        self.kv_store: Dict[UID, Any] = {}

    def get_or_none(
        self, key: UID, proxy_only: bool = False
    ) -> Optional[StorableObject]:
        try:
            return self.get(key=key, proxy_only=proxy_only)
        except KeyError as e:  # noqa: F841
            return None

    def check_collision(self, key: UID) -> None:
        # Check ID collision with pointer's result.
        if self.get_or_none(key=key, proxy_only=True):
            raise Exception(
                "You're not allowed to perform this operation using this ID."
            )

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
            all_values.append(self.get(key=key))

        return all_values

    def __contains__(self, key: StoreKey) -> bool:
        _, key_uid = self.key_to_str_and_uid(key=key)
        return key_uid in self.kv_store

    def resolve_proxy_object(self, obj: Any) -> Any:
        raise Exception(
            f"Proxy Objects {type(obj)} should not be in Nodes with DictStore"
        )

    def __getitem__(self, key: StoreKey) -> StorableObject:
        raise Exception("obj = store[key] not allowed because additional args required")

    def get(self, key: StoreKey, proxy_only: bool = False) -> StorableObject:
        key_str, key_uid = self.key_to_str_and_uid(key=key)
        try:
            store_obj = self.kv_store[key_uid]

            # serialized contents
            if isinstance(store_obj, bytes):
                try:
                    de = deserialize(store_obj, from_bytes=True)
                    obj = de
                except Exception as e:
                    raise Exception(f"Failed to deserialize obj at key {key_str}. {e}")
            else:
                # not serialized
                try:
                    obj = deepcopy(store_obj)
                except Exception as e:
                    raise Exception(
                        f"DictStore should not contain unpickleable objects. {e}"
                    )

            if id(obj) == id(store_obj):
                raise Exception("Objects must use deepcopy or mutation can occur")

            if isinstance(obj, ProxyDataset):
                obj = self.resolve_proxy_object(obj=obj)

            return obj
        except Exception as e:
            print(f"Cant get object {str(key_str)}", e)
            raise KeyError(f"Object not found! for UID: {str(key_str)}")

    def __setitem__(self, key: StoreKey, value: StorableObject) -> None:
        self.set(key=key, value=value)

    def set(self, key: StoreKey, value: StorableObject) -> None:
        _, key_uid = self.key_to_str_and_uid(key=key)
        try:
            store_obj = deepcopy(value)
            self.kv_store[key_uid] = store_obj
            if id(value) == id(store_obj):
                raise Exception("Objects must use deepcopy or mutation can occur")
        except Exception as e:
            # if we get a pickling error from deepcopy we can juse use serialize
            # in theory not having to do this unless necessary means it should be a
            # little faster and less memory intensive
            if "Pickling" in str(e):
                try:
                    # deepcopy falls back to pickle and some objects can't be pickled
                    self.kv_store[key_uid] = serialize(value, to_bytes=True)
                except Exception as ex:
                    raise Exception(f"Failed to serialize object {type(value)}. {ex}")
            else:
                raise Exception(f"Failed to save object {type(value)}. {e}")

    def delete(self, key: UID) -> None:
        key_str, key_uid = self.key_to_str_and_uid(key=key)
        try:
            del self.kv_store[key_uid]
        except Exception as e:
            print(f"{type(self)} Exception in __delitem__ error {key_str}. {e}")

    def clear(self) -> None:
        self.kv_store = {}

    def __repr__(self) -> str:
        return f"{type(self)}"
