# stdlib
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

# third party
from pydantic import BaseSettings

# relative
from ...util import get_fully_qualified_name
from ...util import key_emoji
from ..common.serde.serializable import serializable
from ..common.storeable_object import AbstractStorableObject
from ..common.uid import UID
from .proxy_dataset import ProxyDataset


@serializable(recursive_serde=True)
class StorableObject(AbstractStorableObject):
    __attr_allowlist__ = [
        "id",
        "data",
        "description",
        "tags",
        "read_permissions",
        "search_permissions",
        "write_permissions",
    ]

    """
    StorableObject is a wrapper over some Serializable objects, which we want to keep in an
    ObjectStore. The Serializable objects that we want to store have to be backed up in syft-proto
    in the StorableObject protobuffer, where you can find more details on how to add new types to be
    serialized.

    This object is frozen, you cannot change one in place.

    Arguments:
        id (UID): the id at which to store the data.
        data (Serializable): A serializable object.
        description (Optional[str]): An optional string that describes what you are storing. Useful
        when searching.
        tags (Optional[List[str]]): An optional list of strings that are tags used at search.
        TODO: add docs about read_permission and search_permission

    Attributes:
        id (UID): the id at which to store the data.
        data (Serializable): A serializable object.
        description (Optional[str]): An optional string that describes what you are storing. Useful
        when searching.
        tags (Optional[List[str]]): An optional list of strings that are tags used at search.

    """

    __slots__ = ["id", "_data", "_description", "_tags"]

    def __init__(
        self,
        id: UID,
        data: object,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        read_permissions: Optional[dict] = None,
        search_permissions: Optional[dict] = None,
        write_permissions: Optional[dict] = None,
    ):
        self.id = id
        self.data = data
        self._description: str = description if description else ""
        self._tags: List[str] = tags if tags else []

        # the dict key of "verify key" objects corresponding to people
        # the value is the original request_id to allow lookup later
        # who are allowed to call .get() and download this object.
        self.read_permissions: Dict = read_permissions if read_permissions else {}

        # the dict key of "verify key" objects corresponding to people
        # the value is the original request_id to allow lookup later
        # who are allowed to know that the tensor exists (via search or other means)
        self.search_permissions: dict = search_permissions if search_permissions else {}
        self.write_permissions: dict = write_permissions if write_permissions else {}

    @property
    def object_type(self) -> str:
        return str(type(self.data))

    @property
    def object_qualname(self) -> str:
        return get_fully_qualified_name(self.data)

    # Why define data as a property?
    # For C type/class objects as data.
    # We need to use it's wrapper type very often inside StorableObject, so we set _data
    # attribute as it's wrapper object. But we still want to give a straight API to users,
    # so we return the initial C type object when user call obj.data.
    # For python class objects as data. data and _data are the same thing.
    @property  # type: ignore
    def data(self) -> Any:  # type: ignore
        if type(self._data).__name__.endswith("Wrapper"):
            return self._data.obj
        else:
            return self._data

    @data.setter
    def data(self, value: Any) -> Any:
        if hasattr(value, "_sy_serializable_wrapper_type"):
            self._data = value._sy_serializable_wrapper_type(value=value)
        else:
            self._data = value

    @property
    def is_proxy(self) -> bool:
        return isinstance(self._data, ProxyDataset)

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

    def __repr__(self) -> str:
        return (
            "<Storable: "
            + self.data.__repr__().replace("\n", "").replace("  ", " ")
            + ">"
        )

    @property
    def icon(self) -> str:
        return "🗂️"

    @property
    def pprint(self) -> str:
        output = f"{self.icon} ({self.class_name}) ("
        if hasattr(self.data, "pprint"):
            output += self.data.pprint
        elif self.data is not None:
            output += self.data.__repr__()
        else:
            output += "(Key Only)"
        if len(self._description) > 0:
            output += f" desc: {self.description}"
        if len(self._tags) > 0:
            output += f" tags: {self.tags}"
        if len(self.read_permissions.keys()) > 0:
            output += (
                " can_read: "
                + f"{[key_emoji(key=key) for key in self.read_permissions.keys()]}"
            )

        if len(self.search_permissions.keys()) > 0:
            output += (
                " can_search: "
                + f"{[key_emoji(key=key) for key in self.search_permissions.keys()]}"
            )
        if len(self.write_permissions.keys()) > 0:
            output += (
                " can_write: "
                + f"{[key_emoji(key=key) for key in self.write_permissions.keys()]}"
            )

        output += ")"
        return output

    @property
    def class_name(self) -> str:
        return str(self.__class__.__name__)

    def clean_copy(self, settings: BaseSettings) -> "StorableObject":
        """
        This method return a copy of self, but clean up the search_permissions and
        read_permissions attributes.
        """
        return StorableObject(
            id=self.id, data=self.data, tags=self.tags, description=self.description
        )
