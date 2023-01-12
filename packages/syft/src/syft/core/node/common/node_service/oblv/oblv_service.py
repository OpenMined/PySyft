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
from typing import Type
from typing import Union

# third party
from nacl.signing import VerifyKey
from oblv import OblvClient
import requests

# relative
from ......logger import debug
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
from .oblv_messages import PublishDatasetMessage
from .oblv_messages import PublishDatasetResponse

USER_INPUT_MESSAGES = Union[
    GetPublicKeyMessage,
    PublishDatasetMessage,
    CheckEnclaveConnectionMessage,
    CreateKeyPairMessage,
]

USER_OUTPUT_MESSAGES = Union[SuccessResponseMessage, GetPublicKeyResponse]


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
    Creates a new role in the database.

    Args:
        msg (CreateKeyPairMessage): stores msg address.
        node (DomainInterface): domain node.
        verify_key (VerifyKey): public digital signature/key of the user.

    Raises:
        AuthorizationError: If user does not have permissions to create new key.

    Returns:
        SuccessResponseMessage: Success message on key pair generation.
    """

    # Check if user has permissions to create new roles
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
        private = f_private.read()
        f_private.close()
        f_public = open(file_path + "/" + file_name + "_public.der", "rb")
        public = f_public.read()
        f_public.close()
        debug(type(node))
        node.oblv_keys.remove()
        node.oblv_keys.add_keys(public, private)
        debug(node.oblv_keys.get())
        # return result.stdout.decode('utf-8')
    else:
        raise AuthorizationError("You're not allowed to create a new key pair!")

    return SuccessResponseMessage(
        address=msg.reply_to,
        resp_msg="Success",
    )


def get_public_key_msg(
    msg: GetPublicKeyMessage,
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
    file_name = (
        os.getenv("OBLV_KEY_PATH", "/app/content")
        + "/"
        + os.getenv("OBLV_KEY_NAME", "oblv_key")
        + "_public.der"
    )
    debug("File name : " + file_name)
    try:
        with open(file_name, "rb") as f:
            data = f.read()
        data = encodebytes(data).decode("UTF-8").replace("\n", "")
    except FileNotFoundError:
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
        data = encodebytes(keys.public_key).decode("UTF-8").replace("\n", "")
    except Exception as e:
        print(e)
        raise Exception(e)
    return GetPublicKeyResponse(address=msg.reply_to, response=data)


def publish_dataset(
    msg: PublishDatasetMessage,
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

    if not path.exists(
        os.getenv("OBLV_KEY_PATH", "/app/content")
        + "/"
        + os.getenv("OBLV_KEY_NAME", "oblv_key")
        + "_public.der"
    ):
        create_keys_from_db(node)

    cli = OblvClient(msg.client.token, msg.client.oblivious_user_id)
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
        d = process.stderr.readline().decode()
        debug(d)
        if d.__contains__("Error:  Invalid PCR Values"):
            raise OblvProxyConnectPCRError()
        elif d.lower().__contains__("error"):
            raise OblvEnclaveError(message=d)
        elif d.__contains__("listening on"):
            break

    obj = node.store.get(UID.from_string(msg.dataset_id))
    obj_bytes = serialize(obj.data, to_bytes=True)
    req = requests.post(
        "http://127.0.0.1:3030/tensor/dataset/add",
        files={"input": obj_bytes},
        data={"dataset_id": msg.dataset_id},
    )
    process.kill()
    process.wait(1)
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

    return PublishDatasetResponse(address=msg.reply_to, dataset_id=msg.dataset_id)


def check_connection(
    msg: CheckEnclaveConnectionMessage,
    node: DomainInterface,
    verify_key: VerifyKey,
) -> SuccessResponseMessage:

    """Publish dataset to enclave

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
    _allowed = True

    if _allowed:
        cli = OblvClient(msg.client.token, msg.client.oblivious_user_id)
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
            d = process.stderr.readline().decode()
            debug(d)
            if d.__contains__("Error:  Invalid PCR Values"):
                process.kill()
                process.wait(1)
                raise OblvProxyConnectPCRError()
            elif d.__contains__("Error"):
                process.kill()
                process.wait(1)
                raise OblvEnclaveError(message=d)
            elif d.__contains__("listening on"):
                process.kill()
                process.wait(1)
                break

        debug("Found listening. Now ending the process")

        # To Do - Timeout, and process not found

    else:
        raise AuthorizationError("You're not allowed to test connection!")

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
    if not path.exists(
        os.getenv("OBLV_KEY_PATH", "/app/content")
        + "/"
        + os.getenv("OBLV_KEY_NAME", "oblv_key")
        + "_public.der"
    ):
        create_keys_from_db(node)
    cli = OblvClient(msg.client.token, msg.client.oblivious_user_id)
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
        d = process.stderr.readline().decode()
        debug(d)
        if d.__contains__("Error:  Invalid PCR Values"):
            raise OblvProxyConnectPCRError()
        elif d.lower().__contains__("error"):
            raise OblvEnclaveError(message=d)
        elif d.__contains__("listening on"):
            break

    current_budget = node.users.get_budget_for_user(verify_key)
    data_obj = {
        "publish_request_id": msg.result_id,
        "current_budget": current_budget,
    }
    req = requests.post(
        "http://127.0.0.1:3030/tensor/publish/current_budget", json=data_obj
    )
    process.kill()
    process.wait(1)
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
    if not path.exists(
        os.getenv("OBLV_KEY_PATH", "/app/content")
        + "/"
        + os.getenv("OBLV_KEY_NAME", "oblv_key")
        + "_public.der"
    ):
        create_keys_from_db(node)

    cli = OblvClient(msg.client.token, msg.client.oblivious_user_id)
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
        d = process.stderr.readline().decode()
        debug(d)
        if d.__contains__("Error:  Invalid PCR Values"):
            raise OblvProxyConnectPCRError()
        elif d.lower().__contains__("error"):
            raise OblvEnclaveError(message=d)
        elif d.__contains__("listening on"):
            break
    approval = node.users.deduct_epsilon_for_user(
        verify_key, node.users.get_budget_for_user(verify_key), msg.budget_to_deduct
    )
    req = requests.post(
        "http://127.0.0.1:3030/tensor/publish/budget_deducted",
        json={"publish_request_id": msg.result_id, "budget_deducted": approval},
    )
    process.kill()
    process.wait(1)
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
        PublishDatasetMessage: publish_dataset,
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
            PublishDatasetMessage,
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
