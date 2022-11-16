# stdlib
import csv
import io
import tarfile
from typing import Callable
from typing import Dict as TypeDict
from typing import List
from typing import Type
from typing import Union

# third party
from nacl.signing import VerifyKey
import numpy as np
import torch as th

# relative
from ...... import deserialize
from ......util import get_tracer
from .....common.group import VERIFYALL
from .....common.message import ImmediateSyftMessageWithReply
from .....common.uid import UID
from .....store.storeable_object import StorableObject
from ....domain_interface import DomainInterface
from ...exceptions import AuthorizationError
from ...exceptions import DatasetNotFoundError
from ...node_table.utils import model_to_json
from ..auth import service_auth
from ..node_service import ImmediateNodeServiceWithReply
from ..success_resp_message import SuccessResponseMessage
from .dataset_manager_messages import CreateDatasetMessage
from .dataset_manager_messages import DeleteDatasetMessage
from .dataset_manager_messages import GetDatasetMessage
from .dataset_manager_messages import GetDatasetResponse
from .dataset_manager_messages import GetDatasetsMessage
from .dataset_manager_messages import GetDatasetsResponse
from .dataset_manager_messages import UpdateDatasetMessage

ENCODING = "UTF-8"

tracer = get_tracer()


def _handle_dataset_creation_grid_ui(
    msg: CreateDatasetMessage, node: DomainInterface, verify_key: VerifyKey
) -> None:

    file_obj = io.BytesIO(msg.dataset)
    tar_obj = tarfile.open(fileobj=file_obj)
    tar_obj.extractall()
    dataset_id = node.datasets.register(**msg.metadata)
    for item in tar_obj.getmembers():
        if not item.isdir():
            extracted_file = tar_obj.extractfile(item.name)
            if not extracted_file:
                # TODO: raise CustomError
                raise ValueError("Dataset Tar corrupted")

            reader = csv.reader(
                extracted_file.read().decode().split("\n"),
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
                write_permissions={node.verify_key: node.id, verify_key: None},
            )
            node.store[storable.id] = storable

            node.datasets.add(
                name=item.name,
                dataset_id=str(dataset_id),
                obj_id=str(id_at_location.value),
                dtype=df.__class__.__name__,
                shape=str(tuple(df.shape)),
            )


def _handle_dataset_creation_syft(
    msg: CreateDatasetMessage, node: DomainInterface, verify_key: VerifyKey
) -> None:
    with tracer.start_as_current_span("_handle_dataset_creation_syft"):
        with tracer.start_as_current_span("deserialization"):
            result = deserialize(msg.dataset, from_bytes=True)
        dataset_id = msg.metadata.get("dataset_id")
        if not dataset_id:
            dataset_id = node.datasets.register(**msg.metadata)

        for table_name, table in result.items():
            id_at_location = UID()
            storable = StorableObject(
                id=id_at_location,
                data=table,
                tags=[f"#{table_name}"],
                search_permissions={VERIFYALL: None},
                read_permissions={node.verify_key: node.id, verify_key: None},
                write_permissions={node.verify_key: node.id, verify_key: None},
            )
            with tracer.start_as_current_span("save to DB"):
                node.store[storable.id] = storable

            node.datasets.add(
                name=table_name,
                dataset_id=str(dataset_id),
                obj_id=str(id_at_location.value),
                dtype=str(getattr(table, "dtype", type(table).__name__)),
                shape=str(table.shape),
            )


def create_dataset_msg(
    msg: CreateDatasetMessage,
    node: DomainInterface,
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
    node: DomainInterface,
    verify_key: VerifyKey,
) -> GetDatasetResponse:
    ds, objs = node.datasets.get(msg.dataset_id)
    if not ds:
        raise DatasetNotFoundError
    dataset_json = model_to_json(ds)
    # these types seem broken
    dataset_json["data"] = [
        {"name": obj.name, "id": obj.obj, "dtype": obj.dtype, "shape": obj.shape}  # type: ignore
        for obj in objs
    ]
    return GetDatasetResponse(
        address=msg.reply_to,
        metadata=dataset_json,
    )


def get_all_datasets_metadata_msg(
    msg: GetDatasetsMessage,
    node: DomainInterface,
    verify_key: VerifyKey,
) -> GetDatasetsResponse:
    datasets = []
    for dataset in node.datasets.all():
        ds = model_to_json(dataset)
        _, objs = node.datasets.get(dataset.id)
        # these types seem broken
        ds["data"] = [
            {
                "name": obj.name,  # type: ignore
                "id": obj.obj,  # type: ignore
                "dtype": obj.dtype,  # type: ignore
                "shape": obj.shape,  # type: ignore
            }
            for obj in objs
        ]
        datasets.append(ds)
    return GetDatasetsResponse(
        address=msg.reply_to,
        metadatas=datasets,
    )


def update_dataset_msg(
    msg: UpdateDatasetMessage,
    node: DomainInterface,
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
    msg: DeleteDatasetMessage,
    node: DomainInterface,
    verify_key: VerifyKey,
) -> SuccessResponseMessage:
    _allowed = node.users.can_upload_data(verify_key=verify_key)

    if _allowed:
        # If bin object id exists, then only delete the bin object in the dataset.
        # Otherwise, delete the whole dataset.
        if msg.bin_object_id:
            key = UID(msg.bin_object_id)  # type: ignore
            node.store.delete(key)
        else:
            ds, objs = node.datasets.get(msg.dataset_id)

            if not ds:
                raise DatasetNotFoundError
            # Delete all the bin objects related to the dataset
            for obj in objs:
                node.store.delete(UID(obj.obj))  # type: ignore

            node.datasets.delete(id=msg.dataset_id)
    else:
        raise AuthorizationError("You're not allowed to delete data!")

    return SuccessResponseMessage(
        address=msg.reply_to,
        resp_msg="Dataset deleted successfully!",
    )


class DatasetManagerService(ImmediateNodeServiceWithReply):
    INPUT_TYPE = Union[
        Type[CreateDatasetMessage],
        Type[GetDatasetMessage],
        Type[GetDatasetsMessage],
        Type[UpdateDatasetMessage],
        Type[DeleteDatasetMessage],
    ]

    INPUT_MESSAGES = Union[
        CreateDatasetMessage,
        GetDatasetMessage,
        GetDatasetsMessage,
        UpdateDatasetMessage,
        DeleteDatasetMessage,
    ]

    OUTPUT_MESSAGES = Union[
        SuccessResponseMessage, GetDatasetResponse, GetDatasetsResponse
    ]

    msg_handler_map: TypeDict[INPUT_TYPE, Callable[..., OUTPUT_MESSAGES]] = {
        CreateDatasetMessage: create_dataset_msg,
        GetDatasetMessage: get_dataset_metadata_msg,
        GetDatasetsMessage: get_all_datasets_metadata_msg,
        UpdateDatasetMessage: update_dataset_msg,
        DeleteDatasetMessage: delete_dataset_msg,
    }

    @staticmethod
    @service_auth(guests_welcome=True)
    def process(
        node: DomainInterface,
        msg: INPUT_MESSAGES,
        verify_key: VerifyKey,
    ) -> OUTPUT_MESSAGES:
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
