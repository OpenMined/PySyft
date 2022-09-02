"""
This file defines all the functions/classes to perform oblv actions, for a given domain node, in a RESTful manner.
"""

# stdlib
from base64 import encodebytes
import os
import subprocess
from typing import Callable
from typing import Dict
from typing import List
from typing import Type
from typing import Union

# third party
from cryptography import x509
from nacl.signing import VerifyKey
from oblv import OblvClient
import requests

# relative
from ......grid import GridURL
from ......logger import debug
from .....common.message import ImmediateSyftMessageWithReply
from .....common.uid import UID
from ....common.action.get_object_action import GetObjectAction
from ....common.action.get_object_action import GetObjectResponseMessage
from ....domain_interface import DomainInterface
from ...exceptions import AuthorizationError
from ...exceptions import OblvEnclaveError
from ...exceptions import OblvEnclaveUnAuthorizedError
from ...exceptions import OblvKeyNotFoundError
from ...exceptions import OblvProxyConnectPCRError
from ...exceptions import RequestError
from ...exceptions import RoleNotFoundError
from ...node_table.utils import model_to_json
from ..auth import service_auth
from ..node_service import ImmediateNodeServiceWithReply
from ..node_service import ImmediateNodeServiceWithoutReply
from ..success_resp_message import SuccessResponseMessage
from .oblv_messages import CheckEnclaveConnectionMessage
from .oblv_messages import CreateKeyPairMessage
from .oblv_messages import CreateKeyPairResponse
from .oblv_messages import GetPublicKeyMessage
from .oblv_messages import GetPublicKeyResponse
from .oblv_messages import PublishDatasetMessage

INPUT_TYPE = Union[
    Type[CreateKeyPairMessage],
    Type[GetPublicKeyMessage],
    Type[PublishDatasetMessage]
]

USER_INPUT_MESSAGES = Union[
    GetPublicKeyMessage,
    PublishDatasetMessage,
]

USER_OUTPUT_MESSAGES = Union[SuccessResponseMessage, GetPublicKeyResponse]

def create_key_pair_msg(
    msg: CreateKeyPairMessage,
    node: DomainInterface,
    verify_key: VerifyKey,
) -> SuccessResponseMessage:
    
    """
    Creates a new role in the database.

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
        result = subprocess.run(["/usr/local/bin/oblv", "keygen", "--key-name", os.getenv("OBLV_KEY_NAME", "oblv_key"), "--output", os.getenv("OBLV_KEY_PATH", "/app/content")],capture_output=True)
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

def publish_dataset(msg: PublishDatasetMessage,
    node: DomainInterface,
    verify_key: VerifyKey,
    ) -> SuccessResponseMessage:
    
    """Publish dataset to enclave

    Args:
        msg (PublishDatasetMessage): stores msg address.
        node (DomainInterface): domain node.
        verify_key (VerifyKey): public digital signature/key of the user.

    Raises:
        AuthorizationError: If user does not have permissions to create new role.
        OblvKeyNotFoundError: If no key found.
        OblvProxyConnectPCRError: If unauthorized deployment code used

    Returns:
        SuccessResponseMessage: Success message on key pair generation.
    """

    cli = OblvClient(
        msg.client.token,msg.client.oblivious_user_id
        )
    public_file_name = os.getenv("OBLV_KEY_PATH", "/app/content") + "/" + os.getenv("OBLV_KEY_NAME", "oblv_key") + "_public.der"
    private_file_name = os.getenv("OBLV_KEY_PATH", "/app/content") + "/" + os.getenv("OBLV_KEY_NAME", "oblv_key") + "_private.der"
    depl = cli.deployment_info(msg.deployment_id)
    if depl.is_deleted==True:
        raise OblvEnclaveError("User cannot connect to this deployment, as it is no longer available.")
    process = subprocess.Popen([
        "/usr/local/bin/oblv", "connect",
        "--private-key", private_file_name,
        "--public-key", public_file_name,
        "--url", depl.instance.service_url,
        "--port","443",
        "--lport","3030",
        "--disable-pcr-check"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    while process.poll() is None:
        d = process.stderr.readline().decode()
        debug(d)
        if d.__contains__("Error:  Invalid PCR Values"):
            raise OblvProxyConnectPCRError()
        elif d.__contains__("Error"):
            raise OblvEnclaveError(message=d)
        elif d.__contains__("listening on"):
            break
        
    obj = node.store.get(UID.from_string(msg.dataset_id))
    obj_bytes = obj.data._object2bytes()
    req = requests.post("http://127.0.0.1:3030/tensor/dataset/add", headers={'Content-Type': 'application/octet-stream'}, data=obj_bytes, params={"dataset_name": msg.dataset_id})
    process.kill()
    process.wait(1)
    if req.status_code==401:
        raise OblvEnclaveUnAuthorizedError()
    elif req.status_code!=200:
        raise OblvEnclaveError("Request to publish dataset failed with status {}".format(req.status_code))
    debug("API Called. Now closing")

    return SuccessResponseMessage(
        address=msg.reply_to,
        resp_msg="Success",
    )

def check_connection(msg: CheckEnclaveConnectionMessage,
    node: DomainInterface,
    verify_key: VerifyKey,
    ) -> SuccessResponseMessage:
    
    """Publish dataset to enclave

    Args:
        msg (CheckEnclaveConnectionMessage): stores msg address.
        node (DomainInterface): domain node.
        verify_key (VerifyKey): public digital signature/key of the user.

    Raises:
        AuthorizationError: If user does not have permissions to create new role.
        OblvKeyNotFoundError: If no key found.
        OblvProxyConnectPCRError: If unauthorized deployment code used

    Returns:
        SuccessResponseMessage: Success message on key pair generation.
    """
    _allowed = True

    if _allowed:
        cli = OblvClient(
            msg.client.token,msg.client.oblivious_user_id
            )
        debug("URL = "+cli.deployment_info(msg.deployment_id).instance.service_url)
        public_file_name = os.getenv("OBLV_KEY_PATH", "/app/content") + "/" + os.getenv("OBLV_KEY_NAME", "oblv_key") + "_public.der"
        private_file_name = os.getenv("OBLV_KEY_PATH", "/app/content") + "/" + os.getenv("OBLV_KEY_NAME", "oblv_key") + "_private.der"
        depl = cli.deployment_info(msg.deployment_id)
        if depl.is_deleted==True:
            raise OblvEnclaveError("User cannot connect to this deployment, as it is no longer available.")
        process = subprocess.Popen([
            "/usr/local/bin/oblv", "connect",
            "--private-key", private_file_name,
            "--public-key", public_file_name,
            "--url", depl.instance.service_url,
            "--port","443",
            "--lport","3030",
            "--disable-pcr-check"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        while process.poll() is None:
            d = process.stderr.readline().decode()
            debug(d)
            if d.__contains__("Error:  Invalid PCR Values"):
                raise OblvProxyConnectPCRError()
            elif d.__contains__("Error"):
                raise OblvEnclaveError(message=d)
            elif d.__contains__("listening on"):
                break
        
        debug("Found listening. Now ending the process")
        process.kill()
        process.wait(1)
        
           
        #To Do - Timeout, and process not found
            
        # dataset, objs = node.datasets.get(msg.dataset_id)
        # requests.post("http://127.0.0.1:3030", headers={'Content-Type': 'application/octet-stream'})
        
    else:
        raise AuthorizationError("You're not allowed to test connection!")

    return SuccessResponseMessage(
        address=msg.reply_to,
        resp_msg="Successfully connected to the enclave",
    )
    
    # return


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
            CreateKeyPairMessage
        ]

class OblvRequestUserService(ImmediateNodeServiceWithReply):
    
    msg_handler_map: Dict[type, Callable] = {
        GetPublicKeyMessage: get_public_key_msg,
        PublishDatasetMessage: publish_dataset,
        CheckEnclaveConnectionMessage: check_connection
    }

    @staticmethod
    @service_auth(guests_welcome=True)
    def process(
        node: DomainInterface,
        msg: USER_INPUT_MESSAGES,
        verify_key: VerifyKey,
    ) -> USER_OUTPUT_MESSAGES:
        return OblvRequestUserService.msg_handler_map[type(msg)](
            msg=msg, node=node, verify_key=verify_key
        )

    @staticmethod
    def message_handler_types() -> List[Type[ImmediateSyftMessageWithReply]]:
        return [
            GetPublicKeyMessage, PublishDatasetMessage, CheckEnclaveConnectionMessage
        ]

