"""
This file defines all the functions/classes to perform oblv actions, for a given domain node, in a RESTful manner.
"""

# stdlib
import os
from typing import Callable
from typing import Dict
from typing import List
from typing import Type
from typing import Union
from base64 import encodebytes

# third party
from nacl.signing import VerifyKey
from cryptography import x509

# relative
from ......grid import GridURL
from .....common.message import ImmediateSyftMessageWithReply
from ....domain_interface import DomainInterface
from ...exceptions import AuthorizationError
from ...exceptions import OblvKeyNotFoundError
from ...exceptions import RequestError
from ...exceptions import RoleNotFoundError
from ...node_table.utils import model_to_json
from ..auth import service_auth
from ..node_service import ImmediateNodeServiceWithReply
from ..node_service import ImmediateNodeServiceWithoutReply
from .oblv_messages import CreateKeyPairMessage
from .oblv_messages import GetPublicKeyMessage
from .oblv_messages import CreateKeyPairResponse
from .oblv_messages import GetPublicKeyResponse
from ..success_resp_message import SuccessResponseMessage
from ......logger import debug


import subprocess


INPUT_TYPE = Union[
    Type[CreateKeyPairMessage],
    Type[GetPublicKeyMessage],
]

INPUT_MESSAGES = Union[
    CreateKeyPairMessage,
    GetPublicKeyMessage,
]

OUTPUT_MESSAGES = Union[SuccessResponseMessage, CreateKeyPairResponse, GetPublicKeyResponse]

def create_key_pair_msg(
    msg: CreateKeyPairMessage,
    node: DomainInterface,
    verify_key: VerifyKey,
) -> SuccessResponseMessage:
    """Creates a new role in the database.

    Args:
        msg (CreateKeyPairMessage): stores msg address.
        node (DomainInterface): domain node.
        verify_key (VerifyKey): public digital signature/key of the user.

    Raises:
        AuthorizationError: If user does not have permissions to create new role.

    Returns:
        SuccessResponseMessage: Success message on key pair generation.
    """
    # Check if user has permissions to create new roles
    _allowed = node.users.can_manage_infrastructure(verify_key=verify_key)

    if _allowed:
        result = subprocess.run(["/usr/local/bin/oblv", "keygen", "--key-name", os.getenv("OBLV_KEY_NAME", "oblv_key"), "--output", os.getenv("OBLV_KEY_PATH", "/app/conent")],capture_output=True)
        if result.stderr:
            debug(result.stderr.decode('utf-8'))
            raise subprocess.CalledProcessError(
                    returncode = result.returncode,
                    cmd = result.args,
                    stderr = result.stderr
                    )
        debug(result.stdout.decode('utf-8'))
        # return result.stdout.decode('utf-8')
    else:
        raise AuthorizationError("You're not allowed to create a new key pair!")

    return SuccessResponseMessage(
        address=msg.reply_to,
        resp_msg="Success",
    )

def get_public_key_msg(msg: GetPublicKeyMessage,
    node: DomainInterface,
    verify_key: VerifyKey,
    ) -> SuccessResponseMessage:
    
    """Creates a new role in the database.

    Args:
        msg (CreateKeyPairMessage): stores msg address.
        node (DomainInterface): domain node.
        verify_key (VerifyKey): public digital signature/key of the user.

    Raises:
        AuthorizationError: If user does not have permissions to create new role.
        OblvKeyNotFoundError: If no key found.

    Returns:
        SuccessResponseMessage: Success message on key pair generation.
    """
    # Check if user has permissions to create new roles
    file_name = os.getenv("OBLV_KEY_PATH", "/app/content") + "/" + os.getenv("OBLV_KEY_NAME", "oblv_key") + "_public.der"
    try:
        with open(file_name, "rb") as f:
            data = f.read()
        data = encodebytes(data).decode("UTF-8").replace("\n","")
    except FileNotFoundError:
        raise OblvKeyNotFoundError()
    return GetPublicKeyResponse(
        address=msg.reply_to,
        response=data
    )


class OblvRequestAdminService(ImmediateNodeServiceWithReply):
    
    msg_handler_map: Dict[type, Callable] = {
        CreateKeyPairMessage: create_key_pair_msg,
    }

    @staticmethod
    @service_auth(admin_only=True)
    def process(
        node: DomainInterface,
        msg: Union[CreateKeyPairMessage,GetPublicKeyMessage],
        verify_key: VerifyKey,
    ) -> Union[
        SuccessResponseMessage,
        GetPublicKeyResponse
    ]:
        return OblvRequestAdminService.msg_handler_map[type(msg)](
            msg=msg, node=node, verify_key=verify_key
        )

    @staticmethod
    def message_handler_types() -> List[Type[ImmediateSyftMessageWithReply]]:
        return [
            CreateKeyPairMessage, GetPublicKeyMessage
        ]

class OblvRequestUserService(ImmediateNodeServiceWithReply):
    
    msg_handler_map: Dict[type, Callable] = {
        GetPublicKeyMessage: get_public_key_msg,
    }

    @staticmethod
    @service_auth(guests_welcome=True)
    def process(
        node: DomainInterface,
        msg: GetPublicKeyMessage,
        verify_key: VerifyKey,
    ) -> Union[
        SuccessResponseMessage,
        GetPublicKeyResponse
    ]:
        return OblvRequestUserService.msg_handler_map[type(msg)](
            msg=msg, node=node, verify_key=verify_key
        )

    @staticmethod
    def message_handler_types() -> List[Type[ImmediateSyftMessageWithReply]]:
        return [
            GetPublicKeyMessage
        ]

