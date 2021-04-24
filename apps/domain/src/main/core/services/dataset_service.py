# stdlib
import secrets
from typing import List, Type, Union
from base64 import b64encode, b64decode
from json import dumps

# third party
from nacl.signing import VerifyKey
from nacl.encoding import HexEncoder
from nacl.signing import SigningKey

# syft relative
from syft.core.store import Dataset
from syft.core.common.uid import UID
from syft.core.node.abstract.node import AbstractNode
from syft.core.node.common.service.auth import service_auth
from syft.core.store.storeable_object import StorableObject
from syft.core.node.common.service.node_service import ImmediateNodeServiceWithReply
from syft.core.node.common.service.node_service import ImmediateNodeServiceWithoutReply
from syft.core.common.message import ImmediateSyftMessageWithReply

from syft.grid.messages.dataset_messages import (
    CreateDatasetMessage,
    GetDatasetMessage,
    GetDatasetsMessage,
    UpdateDatasetMessage,
    DeleteDatasetMessage,
    CreateDatasetResponse,
    GetDatasetResponse,
    GetDatasetsResponse,
    UpdateDatasetResponse,
    DeleteDatasetResponse,
)

from ..exceptions import (
    MissingRequestKeyError,
    RoleNotFoundError,
    AuthorizationError,
    UserNotFoundError,
    AuthorizationError,
    DatasetNotFoundError,
)
from ..database.utils import model_to_json
from ..datasets.dataset_ops import (
    create_dataset,
    update_dataset,
    delete_dataset,
    get_specific_dataset_and_relations,
    get_all_relations,
    get_all_datasets,
    get_all_datasets_metadata,
    get_dataset_metadata,
)
from ..database import expand_user_object

ENCODING = "UTF-8"


def create_dataset_msg(
    msg: CreateDatasetMessage,
    node: AbstractNode,
    verify_key: VerifyKey,
) -> CreateDatasetResponse:
    # Get Payload Content
    _current_user_id = msg.content.get("current_user", None)
    users = node.users

    _allowed = users.can_upload_data(user_id=_current_user_id)

    if _allowed:
        _dataset = msg.content.get("dataset", None)
        storage = node.disk_store
        _json = create_dataset(_dataset)
    else:
        raise AuthorizationError("You're not allowed to upload data!")

    return CreateDatasetResponse(
        address=msg.reply_to,
        status_code=200,
        content=_json,
    )


def get_dataset_metadata_msg(
    msg: GetDatasetMessage,
    node: AbstractNode,
    verify_key: VerifyKey,
) -> GetDatasetResponse:
    # Get Payload Content
    _dataset_id = msg.content.get("dataset_id", None)
    _current_user_id = msg.content.get("current_user", None)
    users = node.users
    if not _current_user_id:
        _current_user_id = users.first(
            verify_key=verify_key.encode(encoder=HexEncoder).decode("utf-8")
        ).id

    _msg = {}

    storage = node.disk_store
    ds, objs = get_specific_dataset_and_relations(_dataset_id)
    if not ds:
        raise DatasetNotFoundError
    dataset_json = model_to_json(ds)
    dataset_json["data"] = [
        {"name": obj.name, "id": obj.obj, "dtype": obj.dtype, "shape": obj.shape}
        for obj in objs
    ]

    return GetDatasetResponse(
        address=msg.reply_to,
        status_code=200,
        content=dataset_json,
    )


def get_all_datasets_metadata_msg(
    msg: GetDatasetsMessage,
    node: AbstractNode,
    verify_key: VerifyKey,
) -> GetDatasetsResponse:
    # Get Payload Content
    _current_user_id = msg.content.get("current_user", None)
    users = node.users
    users = node.users
    if not _current_user_id:
        _current_user_id = users.first(
            verify_key=verify_key.encode(encoder=HexEncoder).decode("utf-8")
        ).id

    _msg = {}

    storage = node.disk_store
    datasets = []
    for dataset in get_all_datasets():
        ds = model_to_json(dataset)
        objs = get_all_relations(dataset.id)
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
        status_code=200,
        content=datasets,
    )


def update_dataset_msg(
    msg: UpdateDatasetMessage,
    node: AbstractNode,
    verify_key: VerifyKey,
) -> UpdateDatasetResponse:
    # Get Payload Content
    _current_user_id = msg.content.get("current_user", None)
    _dataset_id = msg.content.get("dataset_id", None)
    _tags = msg.content.get("tags", [])
    _description = msg.content.get("description", "")
    _manifest = msg.content.get("manifest", "")

    users = node.users
    if not _current_user_id:
        _current_user_id = users.first(
            verify_key=verify_key.encode(encoder=HexEncoder).decode("utf-8")
        ).id

    _allowed = users.can_upload_data(user_id=_current_user_id)

    _msg = {}
    if _allowed:
        storage = node.disk_store
        _msg = update_dataset(_dataset_id, _tags, _manifest, _description)
    else:
        raise AuthorizationError("You're not allowed to upload data!")

    return UpdateDatasetResponse(
        address=msg.reply_to,
        status_code=204,
        content={"message": "Dataset updated successfully!"},
    )


def delete_dataset_msg(
    msg: UpdateDatasetMessage,
    node: AbstractNode,
    verify_key: VerifyKey,
) -> DeleteDatasetResponse:
    # Get Payload Content
    _current_user_id = msg.content.get("current_user", None)
    _dataset_id = msg.content.get("dataset_id", None)

    users = node.users
    if not _current_user_id:
        _current_user_id = users.first(
            verify_key=verify_key.encode(encoder=HexEncoder).decode("utf-8")
        ).id

    _allowed = users.can_upload_data(user_id=_current_user_id)

    if _allowed:
        storage = node.disk_store
        delete_dataset(_dataset_id)
    else:
        raise AuthorizationError("You're not allowed to upload data!")

    return DeleteDatasetResponse(
        address=msg.reply_to,
        status_code=204,
        content={"message": "Dataset deleted successfully!"},
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
    ) -> Union[
        CreateDatasetResponse,
        GetDatasetResponse,
        GetDatasetsResponse,
        UpdateDatasetResponse,
        DeleteDatasetResponse,
    ]:
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
