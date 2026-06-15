from dataclasses import dataclass
from enum import IntEnum


class AccessLevel(IntEnum):
    READ = 1
    WRITE = 2
    ADMIN = 4


@dataclass
class User:
    id: str  # email


@dataclass
class ACLRequest:
    path: str  # relative path within the datasite
    level: AccessLevel
    user: User
