# stdlib
from typing import Iterable
from typing import KeysView
from typing import List
from typing import Optional

# third party
from sqlalchemy.orm.decl_api import DeclarativeMeta
from sqlalchemy.orm.session import Session
from torch import Tensor

# syft absolute
import syft
from syft.core.common.uid import UID
from syft.core.store import ObjectStore
from syft.core.store.storeable_object import StorableObject

from ..tables.bin_obj import BinObject
from ..tables.bin_obj_metadata import ObjectMetadata

ENCODING = "UTF-8"


def create_storable(
    _id: UID, data: Tensor, description: str, tags: Iterable[str]
) -> StorableObject:
    obj = StorableObject(id=_id, data=data, description=description, tags=tags)

    return obj


class BinObjectManager(ObjectStore):
    def __init__(
        self,
        db: Session
    ) -> None:
        self.db = db

    def get_object(self, key: UID) -> Optional[StorableObject]:
        try:
            return self.__getitem__(key)
        except KeyError as e:  # noqa: F841
            print(e)
            return None

    def get_objects_of_type(self, obj_type: type) -> Iterable[StorableObject]:
        return [obj for obj in self.values() if isinstance(obj.data, obj_type)]

    def __sizeof__(self) -> int:
        return self.values().__sizeof__()

    def __str__(self) -> str:
        return str(self.values())

    def __len__(self) -> int:
        return self.db.query(ObjectMetadata).count()

    def keys(self) -> KeysView[UID]:
        keys = self.db.query(BinObject.id).all()
        keys = [UID.from_string(k[0]) for k in keys]
        return keys

    def values(self) -> List[StorableObject]:

        obj_keys = self.keys()
        values = []
        for key in obj_keys:
            try:
                values.append(self.__getitem__(key))
            except Exception as e:  # noqa: F841
                print("Unable to get item for key", key)  # TODO: TechDebt add logging
                print(e)
        return values

    def __contains__(self, key: UID) -> bool:
        return (
            self.db.query(BinObject)
            .filter_by(id=str(key.value))
            .first()
            is not None
        )

    def __getitem__(self, key: UID) -> StorableObject:
        bin_obj = (
            self.db.query(BinObject)
            .filter_by(id=str(key.value))
            .first()
        )
        obj_metadata = (
            self.db.query(ObjectMetadata)
            .filter_by(obj=str(key.value))
            .first()
        )

        if not bin_obj or not obj_metadata:
            raise Exception("Object not found!")

        obj = StorableObject(
            id=UID.from_string(bin_obj.id),
            data=bin_obj.object,
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
        return obj

    def __setitem__(self, key: UID, value: StorableObject) -> None:

        bin_obj = BinObject(id=str(key.value), object=value.data)
        # metadata_dict = storable_to_dict(value)
        metadata_obj = ObjectMetadata(
            obj=bin_obj.id,
            tags=value.tags,
            description=value.description,
            read_permissions=syft.serialize(
                syft.lib.python.Dict(value.read_permissions), to_bytes=True
            ).hex(),
            search_permissions=syft.serialize(
                syft.lib.python.Dict(value.search_permissions), to_bytes=True
            ).hex(),
            # name=metadata_dict["name"],
        )

        if self.__contains__(key):
            self.delete(key)

        self.db.add(bin_obj)
        self.db.add(metadata_obj)
        self.db.commit()

    def delete(self, key: UID) -> None:

        try:
            object_to_delete = (
                self.db.query(BinObject)
                .filter_by(id=str(key.value))
                .first()
            )
            metadata_to_delete = (
                self.db.query(BinObject)
                .filter_by(obj=str(key.value))
                .first()
            )
            self.db.delete(object_to_delete)
            self.db.delete(metadata_to_delete)
            self.db.commit()
        except Exception as e:
            print(f"{type(self)} Exception in __delitem__ error {key}. {e}")

    def clear(self) -> None:
        self.db.query(BinObject).delete()
        self.db.query(ObjectMetadata).delete()
        self.db.commit()

    def __repr__(self) -> str:
        return str(self.values())
