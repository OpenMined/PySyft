# stdlib
from enum import Enum
from enum import auto

# relative
from ...serde.serializable import serializable


@serializable(canonical_name="NOTIFIERS", version=1)
class NOTIFIERS(Enum):
    EMAIL = auto()
    SMS = auto()
    SLACK = auto()
    APP = auto()
