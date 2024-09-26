# Can also be specified by the environment variable
# ORCHESTRA_DEPLOYMENT_TYPE
# stdlib
from enum import Enum

# relative
from .serde.serializable import serializable
from .types.syft_object import SYFT_OBJECT_VERSION_1


@serializable()
class DeploymentType(str, Enum):
    __canonical_name__ = "DeploymentType"
    __version__ = SYFT_OBJECT_VERSION_1

    PYTHON = "python"
    REMOTE = "remote"

    def __str__(self) -> str:
        # Use values when transforming ServerType to str
        return self.value
