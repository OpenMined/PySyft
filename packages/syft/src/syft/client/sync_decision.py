from enum import Enum

class SyncDecision(Enum):
    low = "low"
    high = "high"
    skip = "skip"
    ignore = "ignore"