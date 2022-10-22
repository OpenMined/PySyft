# stdlib
from copy import deepcopy
from typing import Any
from typing import Collection
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import cast

# third party
from pydantic import BaseSettings
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm.session import Session

# syft absolute
import syft as sy

# relative
from ....common.uid import UID
from ....node.common.node_table.bin_obj_dataset import BinObjDataset
from ....store import ObjectStore
from ....store.proxy_dataset import ProxyDataset
from ....store.store_interface import StoreKey
from ....store.storeable_object import StorableObject
from ..node_table.bin_obj_metadata import ObjectMetadata


class DictStore(ObjectStore):
    def __init__(self, db: Session, settings: BaseSettings) -> None:
        self.db = db
        self.settings = settings
        self.kv_store: Dict[UID, Any] = {}

    def get_or_none(
        self, key: UID, proxy_only: bool = False
    ) -> Optional[StorableObject]:
        try:
            return self.get(key=key, proxy_only=proxy_only)
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
            local_session = sessionmaker(bind=self.db)()
            store_obj = self.kv_store[key_uid]

            # serialized contents
            if isinstance(store_obj, bytes):
                try:
                    de = sy.deserialize(store_obj, from_bytes=True)
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

            obj_metadata = (
                local_session.query(ObjectMetadata).filter_by(obj=key_str).first()
            )

            if obj is None or obj_metadata is None:
                raise KeyError(f"Object not found! for UID: {key_str}")

            obj = StorableObject(
                id=key_uid,
                data=obj.data,
                description=obj_metadata.description,
                tags=obj_metadata.tags,
                read_permissions=dict(
                    sy.deserialize(
                        bytes.fromhex(obj_metadata.read_permissions), from_bytes=True
                    )
                ),
                search_permissions=dict(
                    sy.deserialize(
                        bytes.fromhex(obj_metadata.search_permissions), from_bytes=True
                    )
                ),
                write_permissions=dict(
                    sy.deserialize(
                        bytes.fromhex(obj_metadata.write_permissions), from_bytes=True
                    )
                ),
            )
            local_session.close()
            return obj
        except Exception as e:
            print(f"Cant get object {str(key_str)}", e)
            raise KeyError(f"Object not found! for UID: {str(key_str)}")

    def is_dataset(self, key: StoreKey) -> bool:
        key_str, _ = self.key_to_str_and_uid(key=key)
        local_session = sessionmaker(bind=self.db)()
        is_dataset_obj = (
            local_session.query(BinObjDataset).filter_by(obj=key_str).exists()
        )
        is_dataset_obj = local_session.query(is_dataset_obj).scalar()
        local_session.close()
        return is_dataset_obj

    def __setitem__(self, key: StoreKey, value: StorableObject) -> None:
        self.set(key=key, value=value)

    def set(self, key: StoreKey, value: StorableObject) -> None:
        key_str, key_uid = self.key_to_str_and_uid(key=key)
        try:
            store_obj = deepcopy(value)
            self.kv_store[key_uid] = store_obj
            if id(value) == id(store_obj):
                raise Exception("Objects must use deepcopy or mutation can occur")
        except Exception as e:
            # if we get a pickling error from deepcopy we can juse use sy.serialize
            # in theory not having to do this unless necessary means it should be a
            # little faster and less memory intensive
            if "Pickling" in str(e):
                try:
                    # deepcopy falls back to pickle and some objects can't be pickled
                    self.kv_store[key_uid] = sy.serialize(value, to_bytes=True)
                except Exception as ex:
                    raise Exception(f"Failed to serialize object {type(value)}. {ex}")
            else:
                raise Exception(f"Failed to save object {type(value)}. {e}")

        create_metadata = True
        local_session = sessionmaker(bind=self.db)()
        try:
            # use existing metadata row to prevent more than 1
            metadata_obj = (
                local_session.query(ObjectMetadata).filter_by(obj=key_str).all()[-1]
            )
            create_metadata = False
        except Exception:
            pass
            # no metadata row exists lets insert one
            metadata_obj = ObjectMetadata()

        metadata_obj.obj = key_str
        metadata_obj.tags = value.tags
        metadata_obj.description = value.description
        metadata_obj.read_permissions = cast(
            bytes,
            sy.serialize(sy.lib.python.Dict(value.read_permissions), to_bytes=True),
        ).hex()
        metadata_obj.search_permissions = cast(
            bytes,
            sy.serialize(sy.lib.python.Dict(value.search_permissions), to_bytes=True),
        ).hex()
        metadata_obj.write_permissions = cast(
            bytes,
            sy.serialize(sy.lib.python.Dict(value.write_permissions), to_bytes=True),
        ).hex()

        if create_metadata:
            local_session.add(metadata_obj)
        local_session.commit()
        local_session.close()

    def delete(self, key: UID) -> None:
        key_str, key_uid = self.key_to_str_and_uid(key=key)
        try:
            del self.kv_store[key_uid]
            local_session = sessionmaker(bind=self.db)()
            metadata_to_delete = (
                local_session.query(ObjectMetadata).filter_by(obj=key_str).first()
            )
            local_session.delete(metadata_to_delete)
            local_session.commit()
            local_session.close()
        except Exception as e:
            print(f"{type(self)} Exception in __delitem__ error {key_str}. {e}")

    def clear(self) -> None:
        self.kv_store = {}
        local_session = sessionmaker(bind=self.db)()
        local_session.query(ObjectMetadata).delete()
        local_session.commit()
        local_session.close()

    def __repr__(self) -> str:
        return f"{type(self)}"
