# stdlib
from typing import List
from typing import Optional

# syft relative
from .serde.serializable import Serializable
from .uid import UID


class AbstractStorableObject(Serializable):

    data: object
    id: UID
    search_permissions: dict

    @property
    def icon(self) -> str:
        return "🗂️"

    @property
    def pprint(self) -> str:
        output = f"{self.icon} ({self.class_name})"
        return output

    @property
    def class_name(self) -> str:
        return str(self.__class__.__name__)

    @property
    def object_type(self) -> str:
        raise NotImplementedError

    @property
    def tags(self) -> Optional[List[str]]:
        raise NotImplementedError

    @property
    def description(self) -> Optional[str]:
        raise NotImplementedError
