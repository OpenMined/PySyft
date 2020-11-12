# syft relative
from .serde.serializable import Serializable
from .uid import UID


class AbstractStorableObject(Serializable):

    data: object
    id: UID

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
