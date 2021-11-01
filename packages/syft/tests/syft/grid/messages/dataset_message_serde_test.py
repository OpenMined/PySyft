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

# from syft.core.node.common.node_service.dataset_manager.dataset_manager_messages import (
#     GetDatasetsResponse,
# )
from syft.core.node.common.node_service.dataset_manager.dataset_manager_messages import (
    CreateDatasetMessage,
)
from syft.core.node.common.node_service.dataset_manager.dataset_manager_messages import (
    DeleteDatasetMessage,
)
from syft.core.node.common.node_service.dataset_manager.dataset_manager_messages import (
    GetDatasetMessage,
)
from syft.core.node.common.node_service.dataset_manager.dataset_manager_messages import (
    GetDatasetResponse,
)
from syft.core.node.common.node_service.dataset_manager.dataset_manager_messages import (
    GetDatasetsMessage,
)
from syft.core.node.common.node_service.dataset_manager.dataset_manager_messages import (
    UpdateDatasetMessage,
)

metadata = {"name": "Dataset", "fields": "age, height, weight"}

test_suite = [
    (
        CreateDatasetMessage,
        {
            "address": Address(),
            "dataset": b"MyBinaryDataset",
            "reply_to": Address(),
            "platform": "syft",
            "metadata": metadata,
        },
        ["address", "dataset", "reply_to", "metadata"],
    ),
    (
        GetDatasetMessage,
        {"address": Address(), "dataset_id": 3, "reply_to": Address()},
        ["address", "dataset_id", "reply_to"],
    ),
    (
        GetDatasetResponse,
        {"address": Address(), "metadata": metadata},
        ["address", "metadata"],
    ),
    (
        GetDatasetsMessage,
        {"address": Address(), "reply_to": Address()},
        ["address", "reply_to"],
    ),
    # (
    #     GetDatasetsResponse,
    #     {
    #         "address": Address(),
    #         "metadatas": [metadata, metadata, metadata],
    #     },
    #     ["address", "metadatas"],
    # ),
    (
        UpdateDatasetMessage,
        {
            "address": Address(),
            "dataset_id": 3,
            "metadata": metadata,
            "reply_to": Address(),
        },
        ["address", "dataset_id", "metadata", "reply_to"],
    ),
    (
        DeleteDatasetMessage,
        {
            "address": Address(),
            "dataset_id": "8f2c411b-fcb7-4059-a0ee-aa370ed03e0c",
            "reply_to": Address(),
        },
        ["address", "dataset_id", "reply_to"],
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
