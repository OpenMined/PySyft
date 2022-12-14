# stdlib
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Sequence

# third party
from sqlalchemy import Column
from sqlalchemy import JSON
from sqlalchemy import String

# relative
from . import Base
from .....lib.python.dict import Dict as SyDict
from ....common.uid import UID
from .syft_object import SyftObject


class Dataset(Base):
    __tablename__ = "dataset"

    id = Column(String(256), primary_key=True)
    name = Column(String(256))
    manifest = Column(String(2048))
    description = Column(String(2048))
    tags = Column(JSON())
    str_metadata = Column(JSON())
    blob_metadata = Column(JSON())


class NoSQLBinObjDataset(SyftObject):
    # version
    __canonical_name__ = "BinObjDataset"
    __version__ = 1

    # fields
    name: str
    obj_id: UID
    dtype: str
    shape: str

    # serde / storage rules
    __attr_state__ = [
        "name",
        "obj_id",
        "dtype",
        "shape",
    ]


class NoSQLDataset(SyftObject):
    # version
    __canonical_name__ = "Dataset"
    __version__ = 1

    # fields
    id: UID
    name: str
    manifest: str
    description: str
    tags: List[str] = []
    str_metadata: Dict
    blob_metadata: Dict
    bin_obj_dataset: List[NoSQLBinObjDataset] = []

    # serde / storage rules
    __attr_state__ = [
        "id",
        "name",
        "manifest" "description" "tags",
        "str_metadata" "blob_metadata",
    ]
    __attr_searchable__: List[str] = []
    __attr_unique__: List[str] = []

    def to_dict(self) -> Dict[Any, Any]:
        attr_dict = super().to_dict()
        del attr_dict["bin_obj_dataset"]
        return attr_dict

    __serde_overrides__: Dict[str, Sequence[Callable]] = {
        "bin_obj_dataset": [
            lambda bin_obj_list: [SyDict(**obj) for obj in bin_obj_list],
            lambda bin_obj_list: [NoSQLBinObjDataset(**obj) for obj in bin_obj_list],
        ]
    }
