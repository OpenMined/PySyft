# stdlib
from ..serde.serializable import serializable
from enum import Enum



@serializable()
class SyncDirection(str, Enum):
    LOW_TO_HIGH = "low_to_high"
    HIGH_TO_LOW = "high_to_low"

class SyncDecision(Enum):
    low = "low"
    high = "high"
    skip = "skip"
    ignore = "ignore"

