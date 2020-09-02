# stdlib
from dataclasses import dataclass
from enum import Enum
from typing import List


class DriveType(Enum):
    HDD = 0
    SSD = 1


@dataclass
class Drive:
    name: str
    storage: int
    drive_type: DriveType = DriveType.SSD


@dataclass
class Storage:
    drives: List[Drive]

    @property
    def storage_total(self) -> int:
        total = 0
        for d in self.drives:
            total += d.storage
        return total

    @property
    def num_drives(self) -> int:
        return len(self.drives)
