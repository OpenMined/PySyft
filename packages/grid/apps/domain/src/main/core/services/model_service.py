# stdlib
from typing import List
from typing import Type
from typing import Union

# third party
import torch as th
from nacl.encoding import HexEncoder
from nacl.signing import VerifyKey
from syft.core.common.message import ImmediateSyftMessageWithReply
from syft.core.node.abstract.node import AbstractNode
from syft.core.node.common.service.auth import service_auth
from syft.core.node.common.service.node_service import ImmediateNodeServiceWithReply
from syft.core.common.uid import UID


# syft relative
from syft.grid.messages.model_messages import DeleteModelMessage
from syft.grid.messages.model_messages import DeleteModelResponse
from syft.grid.messages.model_messages import GetModelMessage
from syft.grid.messages.model_messages import GetModelResponse
from syft.grid.messages.model_messages import GetModelsMessage
from syft.grid.messages.model_messages import GetModelsResponse
from syft.grid.messages.model_messages import UpdateModelMessage
from syft.grid.messages.model_messages import UpdateModelResponse

# grid relative
from ..database.utils import model_to_json
from ..exceptions import AuthorizationError
from ..exceptions import ModelNotFoundError

ENCODING = "UTF-8"


def get_model_metadata_msg(
    msg: GetModelMessage,
    node: AbstractNode,
    verify_key: VerifyKey,
) -> GetModelResponse:
    # Get Payload Content
    _model_id = msg.content.get("model_id", None)
    _current_user_id = msg.content.get("current_user", None)
    users = node.users

    if not _current_user_id:
        _current_user_id = users.first(
            verify_key=verify_key.encode(encoder=HexEncoder).decode("utf-8")
        ).id

    key = UID.from_string(value=_model_id)
    _result = node.store.get_object(key=key)
    if _result and isinstance(_result.data, th.nn.Module):
        model = {
            "id": str(_result.id.value),
            "tags": _result.tags,
            "description": _result.description,
            "name": model.name,
        }
    else:
        raise ModelNotFoundError

    return GetModelResponse(
        address=msg.reply_to,
        status_code=200,
        content=model,
    )


def get_all_models_metadata_msg(
    msg: GetModelsMessage,
    node: AbstractNode,
    verify_key: VerifyKey,
) -> GetModelsResponse:
    # Get Payload Content
    _current_user_id = msg.content.get("current_user", None)
    users = node.users

    if not _current_user_id:
        _current_user_id = users.first(
            verify_key=verify_key.encode(encoder=HexEncoder).decode("utf-8")
        ).id

    _msg = []

    models = node.store.get_objects_of_type(obj_type=th.nn.Module)

    for model in models:
        _msg.append(
            {
                "id": str(model.id.value),
                "tags": model.tags,
                "description": model.description,
                "name": model.name,
            }
        )

    return GetModelsResponse(
        address=msg.reply_to,
        status_code=200,
        content=_msg,
    )


def update_model_msg(
    msg: UpdateModelMessage,
    node: AbstractNode,
    verify_key: VerifyKey,
) -> UpdateModelResponse:
    # Get Payload Content
    _current_user_id = msg.content.get("current_user", None)
    _model_id = msg.content.get("model_id", None)
    _name = msg.content.get("name", None)
    _tags = msg.content.get("tags", [])
    _description = msg.content.get("description", "")

    users = node.users
    if not _current_user_id:
        _current_user_id = users.first(
            verify_key=verify_key.encode(encoder=HexEncoder).decode("utf-8")
        ).id

    _allowed = users.can_upload_data(user_id=_current_user_id)

    key = UID.from_string(value=_model_id)
    _model = node.store.get_object(key=key)

    if _allowed:
        _valid_parameters = _name or _tags or _description
        _valid_model = _model and isinstance(_model.data, th.nn.Module)

        if not (_valid_parameters and _valid_model):
            raise Exception("Invalid parameters!")

        if _tags:
            _model.tags = _tags

        if _description:
            _model.description = _description

        if _name:
            _model.name = _name

        node.store[key] = _model
    else:
        raise AuthorizationError("You're not allowed to upload data!")

    return UpdateModelResponse(
        address=msg.reply_to,
        status_code=204,
        content={"message": "Model updated successfully!"},
    )


def delete_model_msg(
    msg: UpdateModelMessage,
    node: AbstractNode,
    verify_key: VerifyKey,
) -> DeleteModelResponse:
    # Get Payload Content
    _current_user_id = msg.content.get("current_user", None)
    _model_id = msg.content.get("model_id", None)

    users = node.users
    if not _current_user_id:
        _current_user_id = users.first(
            verify_key=verify_key.encode(encoder=HexEncoder).decode("utf-8")
        ).id

    _allowed = users.can_upload_data(user_id=_current_user_id)
    key = UID.from_string(value=_model_id)

    if _allowed:
        node.store.delete(key=key)
    else:
        raise AuthorizationError("You're not allowed to upload data!")

    return DeleteModelResponse(
        address=msg.reply_to,
        status_code=204,
        content={"message": "Model deleted successfully!"},
    )


class ModelManagerService(ImmediateNodeServiceWithReply):

    msg_handler_map = {
        GetModelMessage: get_model_metadata_msg,
        GetModelsMessage: get_all_models_metadata_msg,
        UpdateModelMessage: update_model_msg,
        DeleteModelMessage: delete_model_msg,
    }

    @staticmethod
    @service_auth(guests_welcome=True)
    def process(
        node: AbstractNode,
        msg: Union[
            GetModelMessage,
            GetModelsMessage,
            UpdateModelMessage,
            DeleteModelMessage,
        ],
        verify_key: VerifyKey,
    ) -> Union[
        GetModelResponse,
        GetModelsResponse,
        UpdateModelResponse,
        DeleteModelResponse,
    ]:
        return ModelManagerService.msg_handler_map[type(msg)](
            msg=msg, node=node, verify_key=verify_key
        )

    @staticmethod
    def message_handler_types() -> List[Type[ImmediateSyftMessageWithReply]]:
        return [
            GetModelMessage,
            GetModelsMessage,
            UpdateModelMessage,
            DeleteModelMessage,
        ]
