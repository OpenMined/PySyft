# stdlib
from enum import Enum


class Provider(Enum):
    USER_OWNED = "user-owned"
    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"
    GENESIS = "genesis"
