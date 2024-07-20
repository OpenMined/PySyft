# stdlib
from enum import Enum

# relative
from ..serde.serializable import serializable


@serializable(canonical_name="SyncDirection", version=1)
class SyncDirection(str, Enum):
    LOW_TO_HIGH = "low_to_high"
    HIGH_TO_LOW = "high_to_low"

    def to_sync_decision(self) -> "SyncDecision":
        if self == SyncDirection.LOW_TO_HIGH:
            return SyncDecision.LOW
        elif self == SyncDirection.HIGH_TO_LOW:
            return SyncDecision.HIGH
        else:
            raise ValueError("Invalid SyncDirection")


class SyncDecision(Enum):
    LOW = "low"
    HIGH = "high"
    SKIP = "skip"
    IGNORE = "ignore"
