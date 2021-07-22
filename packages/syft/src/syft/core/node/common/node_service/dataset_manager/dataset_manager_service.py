# stdlib
import csv
import io
import tarfile
from typing import List
from typing import Type
from typing import Union

# third party
from nacl.signing import VerifyKey
import numpy as np
import torch as th

# syft absolute
from syft import deserialize
from syft.core.common.group import VERIFYALL
from syft.core.common.message import ImmediateSyftMessageWithReply
from syft.core.common.uid import UID
from syft.core.node.abstract.node import AbstractNode
from syft.core.node.common.node_service.auth import service_auth
from syft.core.node.common.node_service.node_service import (
    ImmediateNodeServiceWithReply,
)
from syft.core.store.storeable_object import StorableObject
from syft.lib.python import Dict
from syft.lib.python import List as SyftList

# relative
from ...exceptions import AuthorizationError
from ...exceptions import DatasetNotFoundError
from ...node_table.utils import model_to_json
from ..success_resp_message import SuccessResponseMessage
from .dataset_manager_messages import CreateDatasetMessage
from .dataset_manager_messages import DeleteDatasetMessage
from .dataset_manager_messages import GetDatasetMessage
from .dataset_manager_messages import GetDatasetResponse
from .dataset_manager_messages import GetDatasetsMessage
from .dataset_manager_messages import GetDatasetsResponse
from .dataset_manager_messages import UpdateDatasetMessage

ENCODING = "UTF-8"


def _handle_dataset_creation_grid_ui(
    msg: CreateDatasetMessage, node: AbstractNode, verify_key: VerifyKey
) -> SuccessResponseMessage:

    file_obj = io.BytesIO(msg.dataset)
    tar_obj = tarfile.open(fileobj=file_obj)
    tar_obj.extractall()
    dataset_id = node.datasets.register(**msg.metadata)
    for item in tar_obj.members:
        if not item.isdir():
            reader = csv.reader(
                tar_obj.extractfile(item.name).read().decode().split("\n"),
                delimiter=",",
            )
            dataset = []

            for row in reader:
                if len(row) != 0:
                    dataset.append(row)
            dataset = np.array(dataset, dtype=np.float)
            df = th.tensor(dataset, dtype=th.float32)
            id_at_location = UID()

            # Step 2: create message which contains object to send
            storable = StorableObject(
                id=id_at_location,
                data=df,
                tags=["#" + item.name.split("/")[-1]],
                search_permissions={VERIFYALL: None},
                read_permissions={node.verify_key: node.id, verify_key: None},
            )
            node.store[storable.id] = storable

            node.datasets.add(
                name=item.name,
                dataset_id=dataset_id,
                obj_id=str(id_at_location.value),
                dtype=df.__class__.__name__,
                shape=str(tuple(df.shape)),
            )


def _handle_dataset_creation_syft(msg, node, verify_key):
    result = deserialize(msg.dataset, from_bytes=True)
    dataset_id = node.datasets.register(**msg.metadata)

    for table_name, table in result.items():
        id_at_location = UID()

        storable = StorableObject(
            id=id_at_location,
            data=table,
            tags=[f"#{table_name}"],
            search_permissions={VERIFYALL: None},
            read_permissions={node.verify_key: node.id, verify_key: None},
        )
        node.store[storable.id] = storable
        node.datasets.add(
            name=table_name,
            dataset_id=dataset_id,
            obj_id=str(id_at_location.value),
            dtype=str(table.__class__.__name__),
            shape=str(table.shape),
        )


def create_dataset_msg(
    msg: CreateDatasetMessage,
    node: AbstractNode,
    verify_key: VerifyKey,
) -> SuccessResponseMessage:
    # Check key permissions
    _allowed = node.users.can_upload_data(verify_key=verify_key)

    if not _allowed:
        raise AuthorizationError("You're not allowed to upload data!")

    if msg.platform == "syft":
        _handle_dataset_creation_syft(msg, node, verify_key)
    elif msg.platform == "grid-ui":
        _handle_dataset_creation_grid_ui(msg, node, verify_key)

    return SuccessResponseMessage(
        address=msg.reply_to,
        resp_msg="Dataset Created Successfully!",
    )


def get_dataset_metadata_msg(
    msg: GetDatasetMessage,
    node: AbstractNode,
    verify_key: VerifyKey,
) -> GetDatasetResponse:
    ds, objs = node.datasets.get(msg.dataset_id)
    if not ds:
        raise DatasetNotFoundError
    dataset_json = model_to_json(ds)
    dataset_json["data"] = [
        {"name": obj.name, "id": obj.obj, "dtype": obj.dtype, "shape": obj.shape}
        for obj in objs
    ]
    return GetDatasetResponse(
        address=msg.reply_to,
        content=Dict(dataset_json),
    )


def get_all_datasets_metadata_msg(
    msg: GetDatasetsMessage,
    node: AbstractNode,
    verify_key: VerifyKey,
) -> GetDatasetsResponse:
    datasets = []
    for dataset in node.datasets.all():
        ds = model_to_json(dataset)
        _, objs = node.datasets.get(dataset.id)
        ds["data"] = [
            {
                "name": obj.name,
                "id": obj.obj,
                "dtype": obj.dtype,
                "shape": obj.shape,
            }
            for obj in objs
        ]
        datasets.append(ds)
    return GetDatasetsResponse(
        address=msg.reply_to,
        content=SyftList(datasets),
    )


def update_dataset_msg(
    msg: UpdateDatasetMessage,
    node: AbstractNode,
    verify_key: VerifyKey,
) -> SuccessResponseMessage:
    # Get Payload Content
    _allowed = node.users.can_upload_data(verify_key=verify_key)
    if _allowed:
        metadata = {
            key: msg.metadata[key].upcast()
            for (key, value) in msg.metadata.items()
            if msg.metadata[key] is not None
        }

        node.datasets.set(dataset_id=msg.dataset_id, metadata=metadata)
    else:
        raise AuthorizationError("You're not allowed to upload data!")

    return SuccessResponseMessage(
        address=msg.reply_to,
        resp_msg="Dataset updated successfully!",
    )


def delete_dataset_msg(
    msg: UpdateDatasetMessage,
    node: AbstractNode,
    verify_key: VerifyKey,
) -> SuccessResponseMessage:
    _allowed = node.users.can_upload_data(verify_key=verify_key)

    if _allowed:
        node.datasets.delete(id=msg.dataset_id)
    else:
        raise AuthorizationError("You're not allowed to upload data!")

    return SuccessResponseMessage(
        address=msg.reply_to,
        resp_msg="Dataset deleted successfully!",
    )


class DatasetManagerService(ImmediateNodeServiceWithReply):

    msg_handler_map = {
        CreateDatasetMessage: create_dataset_msg,
        GetDatasetMessage: get_dataset_metadata_msg,
        GetDatasetsMessage: get_all_datasets_metadata_msg,
        UpdateDatasetMessage: update_dataset_msg,
        DeleteDatasetMessage: delete_dataset_msg,
    }

    @staticmethod
    @service_auth(guests_welcome=True)
    def process(
        node: AbstractNode,
        msg: Union[
            CreateDatasetMessage,
            GetDatasetMessage,
            GetDatasetsMessage,
            UpdateDatasetMessage,
            DeleteDatasetMessage,
        ],
        verify_key: VerifyKey,
    ) -> Union[SuccessResponseMessage, GetDatasetResponse, GetDatasetsResponse]:
        return DatasetManagerService.msg_handler_map[type(msg)](
            msg=msg, node=node, verify_key=verify_key
        )

    @staticmethod
    def message_handler_types() -> List[Type[ImmediateSyftMessageWithReply]]:
        return [
            CreateDatasetMessage,
            GetDatasetMessage,
            GetDatasetsMessage,
            UpdateDatasetMessage,
            DeleteDatasetMessage,
        ]
