from enum import Enum, auto
from ...serde.serializable import serializable

@serializable()
class NOTIFIERS(Enum):
    EMAIL = auto()
    SMS = auto()
    SLACK = auto()
    APP = auto()