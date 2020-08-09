from .serde.serializable import Serializable
from .uid import UID

class AbstractStorableObject(Serializable):

    data: object
    id: UID