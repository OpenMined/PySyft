# stdlib
import secrets
from typing import List, Type, Union
from base64 import b64encode, b64decode

# third party
from nacl.signing import VerifyKey
from nacl.encoding import HexEncoder
from nacl.signing import SigningKey

# syft relative
from syft.core.node.abstract.node import AbstractNode
from syft.core.node.common.service.auth import service_auth
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
)
from ..database.utils import model_to_json
from ..database import expand_user_object

ENCODING = "UTF-8"


def create_dataset_msg(
    msg: CreateDatasetMessage,
    node: AbstractNode,
) -> CreateDatasetResponse:
    # Get Payload Content
    _current_user_id = msg.content.get("current_user", None)
    users = node.users

    _allowed = users.can_upload_data(user_id=_current_user_id)

    if _allowed:
        _dataset = msg.content.get("dataset", None)
        dataset = b64decode(_dataset)
        storage = node.disk_store
        _id = storage.store_bytes(dataset)
    else:
        raise AuthorizationError("You're not allowed to upload data!")

    return CreateDatasetResponse(
        address=msg.reply_to,
        status_code=200,
        content={_id: _dataset},
    )


def get_dataset_msg(
    msg: GetDatasetMessage,
    node: AbstractNode,
) -> GetDatasetResponse:
    # Get Payload Content
    _dataset_id = msg.content.get("dataset_id", None)
    _current_user_id = msg.content.get("current_user", None)
    users = node.users
    _msg = {}

    _allowed = users.can_triage_requests(user_id=_current_user_id)
    if _allowed:
        storage = node.disk_store
        dataset = storage.get_object(_dataset_id)
        dataset = b64encode(dataset)
        dataset = dataset.decode(ENCODING)
        _msg = {_dataset_id: dataset}
    else:
        raise AuthorizationError("You're not allowed to get a Dataset!")

    return GetDatasetResponse(
        address=msg.reply_to,
        status_code=200,
        content=_msg,
    )


def get_all_datasets_msg(
    msg: GetDatasetsMessage,
    node: AbstractNode,
) -> GetDatasetsResponse:
    # Get Payload Content
    _current_user_id = msg.content.get("current_user", None)
    users = node.users
    _msg = {}

    _allowed = users.can_triage_requests(user_id=_current_user_id)
    if _allowed:
        storage = node.disk_store
        datasets = storage.pairs()
        datasets = {k: b64encode(v).decode(ENCODING) for k, v in datasets.items()}
        _msg = {"datasets": datasets}
    else:
        raise AuthorizationError("You're not allowed to get Datasets!")

    return GetDatasetsResponse(
        address=msg.reply_to,
        status_code=200,
        content=_msg,
    )


def update_dataset_msg(
    msg: UpdateDatasetMessage,
    node: AbstractNode,
) -> UpdateDatasetResponse:
    # Get Payload Content
    _current_user_id = msg.content.get("current_user", None)
    _dataset_id = msg.content.get("dataset_id", None)
    dataset = msg.content.get("dataset", None)

    users = node.users
    _allowed = users.can_upload_data(user_id=_current_user_id)

    if _allowed:
        dataset = b64decode(dataset)
        storage = node.disk_store
        storage.store_bytes_at(_dataset_id, dataset)

    else:
        raise AuthorizationError("You're not allowed to upload data!")

    return UpdateDatasetResponse(
        address=msg.reply_to,
        status_code=204,
        content={},
    )


def delete_dataset_msg(
    msg: UpdateDatasetMessage,
    node: AbstractNode,
) -> DeleteDatasetResponse:
    # Get Payload Content
    _current_user_id = msg.content.get("current_user", None)
    _dataset_id = msg.content.get("dataset_id", None)

    users = node.users
    _allowed = users.can_upload_data(user_id=_current_user_id)

    if _allowed:
        storage = node.disk_store
        storage.delete(_dataset_id)

    else:
        raise AuthorizationError("You're not allowed to upload data!")

    return DeleteDatasetResponse(
        address=msg.reply_to,
        status_code=204,
        content={},
    )


class DatasetManagerService(ImmediateNodeServiceWithReply):

    msg_handler_map = {
        CreateDatasetMessage: create_dataset_msg,
        GetDatasetMessage: get_dataset_msg,
        GetDatasetsMessage: get_all_datasets_msg,
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
        return DatasetManagerService.msg_handler_map[type(msg)](msg=msg, node=node)

    @staticmethod
    def message_handler_types() -> List[Type[ImmediateSyftMessageWithReply]]:
        return [
            CreateDatasetMessage,
            GetDatasetMessage,
            GetDatasetsMessage,
            UpdateDatasetMessage,
            DeleteDatasetMessage,
        ]
