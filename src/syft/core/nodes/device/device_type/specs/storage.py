from enum import Enum
from typing import List


class DriveType(Enum):
    HDD = 0
    SSD = 1


class Drive:
    def __init__(self, name: str, storage: int, drive_type: DriveType = DriveType.SSD):
        self.name = name
        self.storage = storage
        self.drive_type = drive_type


class Storage:
    def __init__(self, drives: List[Drive]):
        self.drives = drives

    @property
    def storage_total(self):
        total = 0
        for d in self.drives:
            total += d.storage
        return total

    @property
    def num_drives(self):
        return len(self.drives)
