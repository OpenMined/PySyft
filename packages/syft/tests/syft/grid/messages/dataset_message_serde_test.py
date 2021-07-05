# stdlib
from typing import Any
from typing import Callable
from typing import Dict
from typing import List

# third party
import pytest

# syft absolute
import syft
from syft import deserialize
from syft import serialize
from syft.core.io.address import Address
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
    GetDatasetsResponse,
)
from syft.core.node.common.node_service.dataset_manager.dataset_manager_messages import (
    UpdateDatasetMessage,
)

metadata = syft.lib.python.Dict(
    {"name": "Dataset", "fields": ["age", "height", "weight"]}
)

test_suite = [
    (
        CreateDatasetMessage,
        {"address": Address(), "dataset": b"MyBinaryDataset", "reply_to": Address()},
        ["address", "dataset", "reply_to"],
    ),
    (
        GetDatasetMessage,
        {"address": Address(), "dataset_id": 123, "reply_to": Address()},
        ["address", "dataset_id", "reply_to"],
    ),
    (
        GetDatasetResponse,
        {"address": Address(), "metadata": metadata, "status_code": 200},
        ["address", "metadata", "status_code"],
    ),
    (
        GetDatasetsMessage,
        {"address": Address(), "reply_to": Address()},
        ["address", "reply_to"],
    ),
    (
        GetDatasetsResponse,
        {
            "address": Address(),
            "datasets_metadata": syft.lib.python.List([metadata, metadata, metadata]),
            "status_code": 200,
        },
        ["address", "datasets_metadata", "status_code"],
    ),
    (
        UpdateDatasetMessage,
        {
            "address": Address(),
            "dataset_id": 123,
            "metadata": metadata,
            "reply_to": Address(),
        },
        ["address", "dataset_id", "metadata", "reply_to"],
    ),
    (
        DeleteDatasetMessage,
        {"address": Address(), "dataset_id": 123, "reply_to": Address()},
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
