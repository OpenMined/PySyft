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
            "source": "127.0.0.1:8082",
            "target": "127.0.0.1:8081",
            "address": Address(),
            "reply_to": Address(),
            "metadata": metadata,
        },
        ["source", "target", "metadata", "address", "reply_to"],
    ),
    (
        ReceiveAssociationRequestMessage,
        {
            "source": "127.0.0.1:8082",
            "target": "127.0.0.1:8081",
            "address": Address(),
            "metadata": metadata,
            "response": "approved",
        },
        ["source", "target", "metadata", "address", "response"],
    ),
    (
        RespondAssociationRequestMessage,
        {
            "source": "127.0.0.1:8082",
            "target": "127.0.0.1:8081",
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
        {
            "address": Address(),
            "content": metadata,
            "source": "127.0.0.1:8082",
            "target": "127.0.0.1:8081",
        },
        ["address", "content", "source", "target"],
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
        {"address": Address(), "content": [metadata, metadata]},
        ["address", "content"],
    ),
]


@pytest.mark.parametrize("msg_constructor, kwargs, test_fields", test_suite)
def test_create(
    msg_constructor: Callable, kwargs: Dict[str, Any], test_fields: List[str]
) -> None:
    msg = msg_constructor(**kwargs)
    regenerated_message = deserialize(serialize(msg))

    for field in test_fields:
        assert getattr(msg, field) == getattr(regenerated_message, field)
