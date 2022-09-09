# stdlib
from typing import Dict
from typing import List

# third party
from sqlalchemy import Column
from sqlalchemy import JSON
from sqlalchemy import String

# relative
from . import Base
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


class NoSQLDataset(SyftObject):
    # version
    __canonical_name__ = "Dataset"
    __version__ = 1

    # fields
    id: UID  # We do not store id in attr_state, as it stores as _id (MongoDB ObjectID)
    name: str
    manifest: str
    description: str
    tags: Dict
    str_metadata: Dict
    blob_metadata: Dict

    # serde / storage rules
    __attr_state__ = [
        "name",
        "manifest" "description" "tags",
        "str_metadata" "blob_metadata",
    ]
    __attr_searchable__: List[str] = []
    __attr_unique__: List[str] = []
