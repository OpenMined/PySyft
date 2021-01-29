# stdlib
from pathlib import Path
import tempfile
from typing import List, Optional

# third party
from loguru import logger
from sqlitedict import SqliteDict
from typing_extensions import Final

# syft relative
from ...decorators import syft_decorator
from ..common.serde.deserialize import _deserialize
from ..common.uid import UID
from .storeable_object import StorableObject


class Dataset:
    @syft_decorator(typechecking=True)
    def __init__(
        self,
        id: UID,
        data: List[object],
        description: str = "",
        tags: Optional[List[str]] = None,
        read_permissions: Optional[dict] = None,
        search_permissions: Optional[dict] = None,
    ):
        self.id = id
        self.data = data
        self._description: str = description
        self._tags: List[str] = tags if tags else []

        # the dict key of "verify key" objects corresponding to people
        # the value is the original request_id to allow lookup later
        # who are allowed to call .get() and download this object.
        self.read_permissions = read_permissions if read_permissions else {}

        # the dict key of "verify key" objects corresponding to people
        # the value is the original request_id to allow lookup later
        # who are allowed to know that the tensor exists (via search or other means)
        self.search_permissions: dict = search_permissions if search_permissions else {}

    @property
    def tags(self) -> Optional[List[str]]:
        return self._tags

    @tags.setter
    def tags(self, value: Optional[List[str]]) -> None:
        self._tags = value if value else []

    @property
    def description(self) -> Optional[str]:
        return self._description

    @description.setter
    def description(self, description: Optional[str]) -> None:
        self._description = description if description else ""

    @staticmethod
    def construct_new_object(
        id: UID,
        data: List[StorableObject],
        description: Optional[str],
        tags: Optional[List[str]],
    ) -> "Dataset":
        return StorableObject(id=id, data=data, description=description, tags=tags)

    @property
    def class_name(self) -> str:
        return str(self.__class__.__name__)
