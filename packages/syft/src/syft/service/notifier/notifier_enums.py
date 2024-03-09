# stdlib
from enum import Enum
from enum import auto

# relative
from ...serde.serializable import serializable


@serializable()
class NOTIFIERS(Enum):
    EMAIL = auto()
    SMS = auto()
    SLACK = auto()
    APP = auto()
