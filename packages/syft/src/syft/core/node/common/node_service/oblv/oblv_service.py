"""
This file defines all the functions/classes to perform oblv actions, for a given domain node, in a RESTful manner.
"""

# stdlib
from base64 import encodebytes
import os
from os import path
import subprocess  # nosec
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Type
from typing import Union

# third party
from nacl.signing import VerifyKey
import requests

# relative
from ......logger import debug
from ......oblv.constants import DOMAIN_CONNECTION_PORT
from ......oblv.constants import LOCAL_MODE
from .....common.message import ImmediateSyftMessageWithReply
from .....common.message import ImmediateSyftMessageWithoutReply
from .....common.serde.serialize import _serialize as serialize
from .....common.uid import UID
from ....domain_interface import DomainInterface
from ...exceptions import AuthorizationError
from ...exceptions import OblvEnclaveError
from ...exceptions import OblvEnclaveUnAuthorizedError
from ...exceptions import OblvProxyConnectPCRError
from ..auth import service_auth
from ..node_service import ImmediateNodeServiceWithReply
from ..node_service import ImmediateNodeServiceWithoutReply
from ..success_resp_message import SuccessResponseMessage
from .oblv_messages import CheckEnclaveConnectionMessage
from .oblv_messages import CreateKeyPairMessage
from .oblv_messages import DeductBudgetMessage
from .oblv_messages import GetPublicKeyMessage
from .oblv_messages import GetPublicKeyResponse
from .oblv_messages import PublishApprovalMessage
from .oblv_messages import TransferDatasetMessage
from .oblv_messages import TransferDatasetResponse

USER_INPUT_MESSAGES = Union[
    GetPublicKeyMessage,
    TransferDatasetMessage,
    CheckEnclaveConnectionMessage,
    CreateKeyPairMessage,
]

USER_OUTPUT_MESSAGES = Union[SuccessResponseMessage, GetPublicKeyResponse]


def make_request_to_enclave(
    node,
    msg,
    request_method: Callable,
    connection_string: str,
    params: Optional[Dict] = None,
    files: Optional[Dict] = None,
    data: Optional[Dict] = None,
    json: Optional[Dict] = None,
):
    if not LOCAL_MODE:
        if not path.exists(
            os.getenv("OBLV_KEY_PATH", "/app/content")
            + "/"
            + os.getenv("OBLV_KEY_NAME", "oblv_key")
            + "_public.der"
        ):
            create_keys_from_db(node)
        cli = msg.oblv_client
        public_file_name = (
            os.getenv("OBLV_KEY_PATH", "/app/content")
            + "/"
            + os.getenv("OBLV_KEY_NAME", "oblv_key")
            + "_public.der"
        )
        private_file_name = (
            os.getenv("OBLV_KEY_PATH", "/app/content")
            + "/"
            + os.getenv("OBLV_KEY_NAME", "oblv_key")
            + "_private.der"
        )
        depl = cli.deployment_info(msg.deployment_id)
        if depl.is_deleted:
            raise OblvEnclaveError(
                "User cannot connect to this deployment, as it is no longer available."
            )
        if depl.is_dev_env:
            process = subprocess.Popen(
                [
                    "/usr/local/bin/oblv",
                    "connect",
                    "--private-key",
                    private_file_name,
                    "--public-key",
                    public_file_name,
                    "--url",
                    depl.instance.service_url,
                    "--pcr0",
                    depl.pcr_codes[0],
                    "--pcr1",
                    depl.pcr_codes[1],
                    "--pcr2",
                    depl.pcr_codes[2],
                    "--port",
                    "443",
                    "--lport",
                    DOMAIN_CONNECTION_PORT,
                    "--disable-pcr-check",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        else:
            process = subprocess.Popen(
                [
                    "/usr/local/bin/oblv",
                    "connect",
                    "--private-key",
                    private_file_name,
                    "--public-key",
                    public_file_name,
                    "--url",
                    depl.instance.service_url,
                    "--pcr0",
                    depl.pcr_codes[0],
                    "--pcr1",
                    depl.pcr_codes[1],
                    "--pcr2",
                    depl.pcr_codes[2],
                    "--port",
                    "443",
                    "--lport",
                    DOMAIN_CONNECTION_PORT,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        while process.poll() is None:
            d = process.stderr.readline().decode()
            debug(d)
            if d.__contains__("Error:  Invalid PCR Values"):
                raise OblvProxyConnectPCRError()
            elif d.lower().__contains__("error"):
                raise OblvEnclaveError(message=d)
            elif d.__contains__("listening on"):
                break
        req = request_method(
            connection_string,
            params=params,
            files=files,
            data=data,
            json=json,
        )
        process.kill()
        process.wait(1)
        return req
    else:
        headers = {"x-oblv-user-name": node.name, "x-oblv-user-role": "domain"}
        return request_method(
            connection_string.replace("127.0.0.1", "host.docker.internal"),
            headers=headers,
            params=params,
            files=files,
            data=data,
            json=json,
        )


def create_keys_from_db(node):
    file_path = os.getenv("OBLV_KEY_PATH", "/app/content")
    file_name = os.getenv("OBLV_KEY_NAME", "oblv_key")
    keys = node.oblv_keys.get()

    # Creating directory if not exist
    os.makedirs(
        os.path.dirname(file_path + "/" + file_name + "_private.der"), exist_ok=True
    )
    f_private = open(file_path + "/" + file_name + "_private.der", "w+b")
    f_private.write(keys.private_key)
    f_private.close()
    f_public = open(file_path + "/" + file_name + "_public.der", "w+b")
    f_public.write(keys.public_key)
    f_public.close()


def create_key_pair_msg(
    msg: CreateKeyPairMessage,
    node: DomainInterface,
    verify_key: VerifyKey,
) -> SuccessResponseMessage:
    """
    Creates a public/private key to be used for Secure Enclave Authentication.

    Args:
        msg (CreateKeyPairMessage): stores msg address.
        node (DomainInterface): domain node.
        verify_key (VerifyKey): public digital signature/key of the user.

    Raises:
        AuthorizationError: If user does not have permissions to create new key.

    Returns:
        SuccessResponseMessage: Success message on key pair generation.
    """
    # Check if user has permissions to create new public/private key pair
    _allowed = node.users.can_manage_infrastructure(verify_key=verify_key)
    file_path = os.getenv("OBLV_KEY_PATH", "/app/content")
    file_name = os.getenv("OBLV_KEY_NAME", "oblv_key")
    if _allowed:
        result = subprocess.run(  # nosec
            [
                "/usr/local/bin/oblv",
                "keygen",
                "--key-name",
                file_name,
                "--output",
                file_path,
            ],
            capture_output=True,
        )
        if result.stderr:
            debug(result.stderr.decode("utf-8"))
            raise subprocess.CalledProcessError(  # nosec
                returncode=result.returncode, cmd=result.args, stderr=result.stderr
            )
        debug(result.stdout.decode("utf-8"))
        f_private = open(file_path + "/" + file_name + "_private.der", "rb")
        private_key = f_private.read()
        f_private.close()
        f_public = open(file_path + "/" + file_name + "_public.der", "rb")
        public_key = f_public.read()
        f_public.close()
        debug(type(node))
        node.oblv_keys.remove()
        node.oblv_keys.add_keys(public_key=public_key, private_key=private_key)
        debug(node.oblv_keys.get())
        # return result.stdout.decode('utf-8')
    else:
        raise AuthorizationError("You're not allowed to create a new key pair!")

    return SuccessResponseMessage(
        address=msg.reply_to,
        resp_msg=f"Successfully created a new public/private key pair on the domain node: {node.name}",
    )


def get_public_key_msg(
    msg: GetPublicKeyMessage,
    node: DomainInterface,
    verify_key: VerifyKey,
) -> GetPublicKeyResponse:

    """Retrieves the oblv public_key from the database.

    Args:
        msg (CreateKeyPairMessage): stores msg address.
        node (DomainInterface): domain node.
        verify_key (VerifyKey): public digital signature/key of the user.

    Raises:
        OblvKeyNotFoundError: If no key found.

    Returns:
        GetPublicKeyResponse: Public Key response message.
    """
    keys = node.oblv_keys.get()
    public_key_str = encodebytes(keys.public_key).decode("UTF-8").replace("\n", "")

    return GetPublicKeyResponse(address=msg.reply_to, response=public_key_str)


def transfer_dataset(
    msg: TransferDatasetMessage,
    node: DomainInterface,
    verify_key: VerifyKey,
) -> TransferDatasetResponse:
    """Transfer dataset to enclave

    Args:
        msg (PublishDatasetMessage): stores msg address.
        node (DomainInterface): domain node.
        verify_key (VerifyKey): public digital signature/key of the user.

    Raises:
        OblvKeyNotFoundError: If no key found.
        OblvProxyConnectPCRError: If unauthorized deployment code used

    Returns:
        TransferDatasetResponse: Response Message after transfer of dataset.
    """
    obj = node.store.get(UID.from_string(msg.dataset_id))
    obj_bytes = serialize(obj.data, to_bytes=True)

    req = make_request_to_enclave(
        node,
        msg,
        requests.post,
        connection_string=f"http://127.0.0.1:{DOMAIN_CONNECTION_PORT}/tensor/dataset/add",
        files={"input": obj_bytes},
        data={"dataset_id": msg.dataset_id},
    )

    if req.status_code == 401:
        raise OblvEnclaveUnAuthorizedError()
    elif req.status_code == 400:
        raise OblvEnclaveError(req.json()["detail"])
    elif req.status_code == 422:
        debug(req.text)
    elif req.status_code != 200:
        raise OblvEnclaveError(
            "Request to publish dataset failed with status {}".format(req.status_code)
        )
    debug("API Called. Now closing")

    return TransferDatasetResponse(address=msg.reply_to, dataset_id=msg.dataset_id)


def check_connection(
    msg: CheckEnclaveConnectionMessage,
    node: DomainInterface,
    verify_key: VerifyKey,
) -> SuccessResponseMessage:
    """Checks if domain node could connect to the provisioned enclave.

    Args:
        msg (CheckEnclaveConnectionMessage): stores msg address.
        node (DomainInterface): domain node.
        verify_key (VerifyKey): public digital signature/key of the user.

    Raises:
        OblvKeyNotFoundError: If no key found.
        OblvProxyConnectPCRError: If unauthorized deployment code used

    Returns:
        SuccessResponseMessage: Success message on key pair generation.
    """
    cli = msg.oblv_client
    debug("URL = " + cli.deployment_info(msg.deployment_id).instance.service_url)
    public_file_name = (
        os.getenv("OBLV_KEY_PATH", "/app/content")
        + "/"
        + os.getenv("OBLV_KEY_NAME", "oblv_key")
        + "_public.der"
    )
    private_file_name = (
        os.getenv("OBLV_KEY_PATH", "/app/content")
        + "/"
        + os.getenv("OBLV_KEY_NAME", "oblv_key")
        + "_private.der"
    )
    depl = cli.deployment_info(msg.deployment_id)
    if depl.is_deleted:
        raise OblvEnclaveError(
            "User cannot connect to this deployment, as it is no longer available."
        )
    if depl.is_dev_env:
        process = subprocess.Popen(  # nosec
            [
                "/usr/local/bin/oblv",
                "connect",
                "--private-key",
                private_file_name,
                "--public-key",
                public_file_name,
                "--url",
                depl.instance.service_url,
                "--pcr0",
                depl.pcr_codes[0],
                "--pcr1",
                depl.pcr_codes[1],
                "--pcr2",
                depl.pcr_codes[2],
                "--port",
                "443",
                "--lport",
                "3030",
                "--disable-pcr-check",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    else:
        process = subprocess.Popen(  # nosec
            [
                "/usr/local/bin/oblv",
                "connect",
                "--private-key",
                private_file_name,
                "--public-key",
                public_file_name,
                "--url",
                depl.instance.service_url,
                "--pcr0",
                depl.pcr_codes[0],
                "--pcr1",
                depl.pcr_codes[1],
                "--pcr2",
                depl.pcr_codes[2],
                "--port",
                "443",
                "--lport",
                "3030",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    while process.poll() is None:
        log_line = process.stderr.readline().decode()
        if "Error:  Invalid PCR Values" in log_line:
            process.kill()
            process.wait(1)
            raise OblvProxyConnectPCRError()
        elif "error" in log_line.lower():
            process.kill()
            process.wait(1)
            raise OblvEnclaveError(message=log_line)
        elif "listening on" in log_line:
            process.kill()
            process.wait(1)
            break

    debug("Found listening. Now ending the process")

    # To Do - Timeout, and process not found

    return SuccessResponseMessage(
        address=msg.reply_to,
        resp_msg="Successfully connected to the enclave",
    )


def dataset_publish_budget(
    msg: PublishApprovalMessage,
    node: DomainInterface,
    verify_key: VerifyKey,
):
    """Provide approval for dataset publish

    Args:
        msg (PublishApprovalMessage): stores msg address.
        node (DomainInterface): domain node.
        verify_key (VerifyKey): public digital signature/key of the user.

    Raises:
        OblvKeyNotFoundError: If no key found.
        OblvProxyConnectPCRError: If unauthorized deployment code used

    Returns:
        SuccessResponseMessage: Success message on key pair generation.
    """
    current_budget = node.users.get_budget_for_user(verify_key)
    data_obj = {
        "publish_request_id": msg.result_id,
        "current_budget": current_budget,
    }
    req = make_request_to_enclave(
        node,
        msg,
        requests.post,
        connection_string=f"http://127.0.0.1:{DOMAIN_CONNECTION_PORT}/tensor/publish/current_budget",
        json=data_obj,
    )

    debug(req.text)
    if req.status_code == 401:
        raise OblvEnclaveUnAuthorizedError()
    elif req.status_code == 400:
        raise OblvEnclaveError(req.json()["detail"])
    elif req.status_code == 422:
        debug(req.text)
    elif req.status_code != 200:
        raise OblvEnclaveError(
            "Request to publish dataset failed with status {}".format(req.status_code)
        )


def dataset_publish_budget_deduction(
    msg: DeductBudgetMessage, node: DomainInterface, verify_key: VerifyKey
):
    """Deduct budget for dataset publish

    Args:
        msg (DeductBudgetMessage): stores msg address.
        node (DomainInterface): domain node.
        verify_key (VerifyKey): public digital signature/key of the user.

    Raises:
        OblvKeyNotFoundError: If no key found.
        OblvProxyConnectPCRError: If unauthorized deployment code used

    Returns:
        SuccessResponseMessage: Success message on key pair generation.
    """
    approval = node.users.deduct_epsilon_for_user(
        verify_key, node.users.get_budget_for_user(verify_key), msg.budget_to_deduct
    )
    req = make_request_to_enclave(
        node,
        msg,
        requests.post,
        connection_string=f"http://127.0.0.1:{DOMAIN_CONNECTION_PORT}/tensor/publish/budget_deducted",
        json={"publish_request_id": msg.result_id, "budget_deducted": approval},
    )
    if req.status_code == 401:
        raise OblvEnclaveUnAuthorizedError()
    elif req.status_code == 400:
        raise OblvEnclaveError(req.json()["detail"])
    elif req.status_code == 422:
        debug(req.text)
    elif req.status_code != 200:
        raise OblvEnclaveError(
            "Request to publish dataset failed with status {}".format(req.status_code)
        )
    else:
        if req.json() != "Success":
            debug("Already deducted so updating again")
            user = node.users.get_user(verify_key=verify_key)
            node.users.set(user_id=user.id, budget=user.budget + msg.budget_to_deduct)
    return "Success"


class OblvRequestAdminService(ImmediateNodeServiceWithReply):

    msg_handler_map: Dict[type, Callable] = {
        CreateKeyPairMessage: create_key_pair_msg,
    }

    @staticmethod
    @service_auth(admin_only=True)
    def process(
        node: DomainInterface,
        msg: Union[CreateKeyPairMessage, GetPublicKeyMessage],
        verify_key: VerifyKey,
    ) -> Union[SuccessResponseMessage, GetPublicKeyResponse]:
        return OblvRequestAdminService.msg_handler_map[type(msg)](
            msg=msg, node=node, verify_key=verify_key
        )

    @staticmethod
    def message_handler_types() -> List[Type[ImmediateSyftMessageWithReply]]:
        return [CreateKeyPairMessage]


class OblvRequestUserService(ImmediateNodeServiceWithReply):

    msg_handler_map: Dict[type, Callable] = {
        GetPublicKeyMessage: get_public_key_msg,
        TransferDatasetMessage: transfer_dataset,
        CheckEnclaveConnectionMessage: check_connection,
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
            GetPublicKeyMessage,
            TransferDatasetMessage,
            CheckEnclaveConnectionMessage,
        ]


class OblvBackgroundService(ImmediateNodeServiceWithoutReply):
    msg_handler_map: Dict[type, Callable] = {
        PublishApprovalMessage: dataset_publish_budget,
        DeductBudgetMessage: dataset_publish_budget_deduction,
    }

    @staticmethod
    @service_auth(guests_welcome=True)
    def process(
        node: DomainInterface,
        msg: Union[PublishApprovalMessage, DeductBudgetMessage],
        verify_key: VerifyKey,
    ) -> USER_OUTPUT_MESSAGES:
        return OblvBackgroundService.msg_handler_map[type(msg)](
            msg=msg, node=node, verify_key=verify_key
        )

    @staticmethod
    def message_handler_types() -> List[Type[ImmediateSyftMessageWithoutReply]]:
        return [PublishApprovalMessage, DeductBudgetMessage]
