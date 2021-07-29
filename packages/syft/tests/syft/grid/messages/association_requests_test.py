# stdlib
from typing import Any
from typing import Callable
from typing import Dict
from typing import List

# third party
import pytest

# syft absolute
from syft import deserialize
from syft import serialize
from syft.core.io.address import Address
from syft.core.node.abstract.node import AbstractNodeClient
from syft.core.node.common.node_service.association_request.association_request_service import (
    DeleteAssociationRequestMessage,
)
from syft.core.node.common.node_service.association_request.association_request_service import (
    GetAssociationRequestMessage,
)
from syft.core.node.common.node_service.association_request.association_request_service import (
    GetAssociationRequestResponse,
)
from syft.core.node.common.node_service.association_request.association_request_service import (
    GetAssociationRequestsMessage,
)
from syft.core.node.common.node_service.association_request.association_request_service import (
    GetAssociationRequestsResponse,
)
from syft.core.node.common.node_service.association_request.association_request_service import (
    ReceiveAssociationRequestMessage,
)
from syft.core.node.common.node_service.association_request.association_request_service import (
    RespondAssociationRequestMessage,
)
from syft.core.node.common.node_service.association_request.association_request_service import (
    SendAssociationRequestMessage,
)

metadata = {"email": "tudor@openmined.org", "reason": "I wanna be cool!"}

test_suite = [
    (
        SendAssociationRequestMessage,
        {
            "source": None,
            "target": None,
            "address": Address(),
            "reply_to": Address(),
            "metadata": metadata,
        },
        ["source", "target", "metadata", "address", "reply_to"],
    ),
    (
        ReceiveAssociationRequestMessage,
        {
            "source": None,
            "target": None,
            "address": Address(),
            "reply_to": Address(),
            "metadata": metadata,
            "response": "approved",
        },
        ["source", "target", "metadata", "address", "reply_to", "response"],
    ),
    (
        RespondAssociationRequestMessage,
        {
            "source": None,
            "target": None,
            "reply_to": Address(),
            "address": Address(),
            "response": "deny",
        },
        ["source", "target", "address", "reply_to", "response"],
    ),
    (
        GetAssociationRequestMessage,
        {"reply_to": Address(), "address": Address(), "association_id": 1},
        ["reply_to", "address", "association_id"],
    ),
    (
        GetAssociationRequestResponse,
        {"address": Address(), "metadata": metadata, "source": None, "target": None},
        ["address", "metadata", "source", "target"],
    ),
    (
        DeleteAssociationRequestMessage,
        {"address": Address(), "association_id": 1, "reply_to": Address()},
        ["address", "reply_to", "association_id"],
    ),
    (
        GetAssociationRequestsMessage,
        {"address": Address(), "reply_to": Address()},
        ["address", "reply_to"],
    ),
    (
        GetAssociationRequestsResponse,
        {"address": Address(), "metadatas": [metadata, metadata]},
        ["address", "metadatas"],
    ),
]


@pytest.mark.parametrize("msg_constructor, kwargs, test_fields", test_suite)
def test_create(
    msg_constructor: Callable,
    kwargs: Dict[str, Any],
    test_fields: List[str],
    client: AbstractNodeClient,
) -> None:
    if "source" in kwargs:
        kwargs["source"] = client

    if "target" in kwargs:
        kwargs["target"] = client

    msg = msg_constructor(**kwargs)
    regenerated_message = deserialize(serialize(msg))

    for field in test_fields:
        assert getattr(msg, field) == getattr(regenerated_message, field)
