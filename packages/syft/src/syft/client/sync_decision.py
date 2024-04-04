# stdlib
from enum import Enum

# relative
from ..serde.serializable import serializable


@serializable()
class SyncDirection(str, Enum):
    LOW_TO_HIGH = "low_to_high"
    HIGH_TO_LOW = "high_to_low"


class SyncDecision(Enum):
    low = "low"
    high = "high"
    skip = "skip"
    ignore = "ignore"
