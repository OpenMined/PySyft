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
)
from ..database.utils import model_to_json
from ..datasets.dataset_ops import (
    create_dataset,
    update_dataset,
    delete_dataset,
    get_all_datasets_metadata,
    get_dataset_metadata,
)
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
) -> GetDatasetResponse:
    # Get Payload Content
    _dataset_id = msg.content.get("dataset_id", None)
    _current_user_id = msg.content.get("current_user", None)
    users = node.users
    _msg = {}

    _allowed = users.can_triage_requests(user_id=_current_user_id)
    if _allowed:
        storage = node.disk_store
        _msg = get_dataset_metadata(_dataset_id)
    else:
        raise AuthorizationError("You're not allowed to get a Dataset!")

    return GetDatasetResponse(
        address=msg.reply_to,
        status_code=200,
        content=_msg,
    )


def get_all_datasets_metadata_msg(
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
        _msg = get_all_datasets_metadata()
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

    _msg = {}
    if _allowed:
        storage = node.disk_store
        _msg = update_dataset(_dataset_id, dataset)

    else:
        raise AuthorizationError("You're not allowed to upload data!")

    return UpdateDatasetResponse(
        address=msg.reply_to,
        status_code=204,
        content=_msg,
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
        delete_dataset(_dataset_id)

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
