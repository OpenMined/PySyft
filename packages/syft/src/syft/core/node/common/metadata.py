# relative
from ...common.serde.serializable import serializable
from ...common.uid import UID


@serializable(recursive_serde=True)
class Metadata:
    __attr_allowlist__ = [
        "name",
        "id",
        "node_type",
        "version",
        "deployed_on",
        "organization",
        "description",
    ]

    def __init__(
        self,
        # node: Location,
        id: UID,
        name: str = "",
        node_type: str = "",
        version: str = "",
        deployed_on: str = "",
        organization: str = "",
        description: str = "",
    ) -> None:
        super().__init__()
        self.name = name
        self.id = id
        if not isinstance(id, UID):
            raise Exception(f"Must have a id of type UID instead got {id}: {type(id)}")
        self.node_type = node_type
        self.version = version
        self.deployed_on = deployed_on
        self.description = description
        self.organization = organization
