# stdlib
from enum import Enum


class UserApplicationStatus(Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
