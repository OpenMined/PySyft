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

ENCODING = "UTF-8"


def create_storable(
    _id: UID, data: Tensor, description: str, tags: Iterable[str]
) -> StorableObject:
    obj = StorableObject(id=_id, data=data, description=description, tags=tags)

    return obj


class BinObjectManager(ObjectStore):
    def __init__(
        self,
        db_session: Session,
        bin_obj_table: DeclarativeMeta,
        bin_obj_metadata_table: DeclarativeMeta,
    ) -> None:
        self.db_session = db_session
        self.bin_obj_table = bin_obj_table
        self.bin_obj_metadata_table = bin_obj_metadata_table

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
        return self.db_session.query(self.bin_obj_metadata_table).count()

    def keys(self) -> KeysView[UID]:
        keys = self.db_session.query(self.bin_obj_table.id).all()
        keys = [UID.from_string(k[0]) for k in keys]
        return keys

    def values(self) -> List[StorableObject]:
        print("Getting all of those values")
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
            self.db_session.query(self.bin_obj_table)
            .filter_by(id=str(key.value))
            .first()
            is not None
        )

    def __getitem__(self, key: UID) -> StorableObject:
        bin_obj = (
            self.db_session.query(self.bin_obj_table)
            .filter_by(id=str(key.value))
            .first()
        )
        obj_metadata = (
            self.db_session.query(self.bin_obj_metadata_table)
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
        print("Storing object:" + str(key) + " -> " + str(value))
        bin_obj = self.bin_obj_table(id=str(key.value), object=value.data)
        # metadata_dict = storable_to_dict(value)
        metadata_obj = self.bin_obj_metadata_table(
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

        self.db_session.add(bin_obj)
        self.db_session.add(metadata_obj)
        self.db_session.commit()

    def delete(self, key: UID) -> None:
        print("Deleting object:" + str(key))
        try:
            object_to_delete = (
                self.db_session.query(self.bin_obj_table)
                .filter_by(id=str(key.value))
                .first()
            )
            metadata_to_delete = (
                self.db_session.query(self.bin_obj_metadata_table)
                .filter_by(obj=str(key.value))
                .first()
            )
            self.db_session.delete(object_to_delete)
            self.db_session.delete(metadata_to_delete)
            self.db_session.commit()
        except Exception as e:
            print(f"{type(self)} Exception in __delitem__ error {key}. {e}")

    def clear(self) -> None:
        self.db_session.query(self.bin_obj_table).delete()
        self.db_session.query(self.bin_obj_metadata_table).delete()
        self.db_session.commit()

    def __repr__(self) -> str:
        return str(self.values())
