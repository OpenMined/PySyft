# stdlib
from copy import deepcopy
from typing import Any
from typing import Collection
from typing import Iterable
from typing import List
from typing import Optional
from typing import Union
from typing import cast

# third party
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm.session import Session

# syft absolute
import syft as sy

# relative
from ....common.uid import UID
from ....node.common.node_table.bin_obj_dataset import BinObjDataset
from ....store import ObjectStore
from ....store.storeable_object import StorableObject
from ..node_table.bin_obj_metadata import ObjectMetadata


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

            if isinstance(key, UID):
                key_str = str(key.value)
                key_uid = key
            elif isinstance(key, bytes):
                key_str = str(key.decode("utf-8"))
                key_uid = UID.from_string(key_str)
            else:
                key_str = key
                key_uid = UID.from_string(key_str)

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
            print(f"Cant get object {str(key)}", e)
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
        key_str = str(key.value)
        try:
            store_obj = deepcopy(value)
            self.kv_store[key] = store_obj
            if id(value) == id(store_obj):
                raise Exception("Objects must use deepcopy or mutation can occur")
        except Exception as e:
            # if we get a pickling error from deepcopy we can juse use sy.serialize
            # in theory not having to do this unless necessary means it should be a
            # little faster and less memory intensive
            if "Pickling" in str(e):
                try:
                    # deepcopy falls back to pickle and some objects can't be pickled
                    self.kv_store[key] = sy.serialize(value, to_bytes=True)
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

        metadata_obj.obj = str(key.value)
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

        if create_metadata:
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
