# stdlib
from typing import Iterable
from typing import KeysView
from typing import List
from typing import Optional

# third party
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm.session import Session
from torch import Tensor

# syft absolute
import syft
from syft.core.common.uid import UID
from syft.core.store import ObjectStore
from syft.core.store.storeable_object import StorableObject

# relative
from ..node_table.bin_obj import BinObject
from ..node_table.bin_obj_metadata import ObjectMetadata

ENCODING = "UTF-8"


def create_storable(
    _id: UID, data: Tensor, description: str, tags: Iterable[str]
) -> StorableObject:
    obj = StorableObject(id=_id, data=data, description=description, tags=tags)

    return obj


class BinObjectManager(ObjectStore):
    def __init__(self, db: Session) -> None:
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
        local_session = sessionmaker(bind=self.db)()
        result = local_session.query(ObjectMetadata).count()
        local_session.close()
        return result

    def keys(self) -> KeysView[UID]:
        local_session = sessionmaker(bind=self.db)()
        keys = local_session.query(BinObject.id).all()
        keys = [UID.from_string(k[0]) for k in keys]
        local_session.close()
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
        local_session = sessionmaker(bind=self.db)()
        result = (
            local_session.query(BinObject).filter_by(id=str(key.value)).first()
            is not None
        )
        local_session.close()
        return result

    def __getitem__(self, key: UID) -> StorableObject:
        local_session = sessionmaker(bind=self.db)()
        bin_obj = local_session.query(BinObject).filter_by(id=str(key.value)).first()
        obj_metadata = (
            local_session.query(ObjectMetadata).filter_by(obj=str(key.value)).first()
        )

        if not bin_obj or not obj_metadata:
            raise Exception("Object not found!")

        obj = StorableObject(
            id=UID.from_string(bin_obj.id),
            data=bin_obj.obj,
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

    def __setitem__(self, key: UID, value: StorableObject) -> None:

        bin_obj = BinObject(id=str(key.value), obj=value.data)
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

        local_session = sessionmaker(bind=self.db)()
        local_session.add(bin_obj)
        local_session.add(metadata_obj)
        local_session.commit()
        local_session.close()

    def delete(self, key: UID) -> None:
        try:
            local_session = sessionmaker(bind=self.db)()

            object_to_delete = (
                local_session.query(BinObject).filter_by(id=str(key.value)).first()
            )
            metadata_to_delete = (
                local_session.query(ObjectMetadata)
                .filter_by(obj=str(key.value))
                .first()
            )

            local_session.delete(metadata_to_delete)
            local_session.delete(object_to_delete)
            local_session.commit()
            local_session.close()
        except Exception as e:
            print(f"{type(self)} Exception in __delitem__ error {key}. {e}")

    def clear(self) -> None:
        local_session = sessionmaker(bind=self.db)()
        local_session.query(BinObject).delete()
        local_session.query(ObjectMetadata).delete()
        local_session.commit()
        local_session.close()

    def __repr__(self) -> str:
        return str(self.values())
