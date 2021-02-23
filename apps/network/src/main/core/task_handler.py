from .codes import RESPONSE_MSG
from json.decoder import JSONDecodeError
from flask_executor import Executor
from flask import request

from syft.core.common.message import SignedImmediateSyftMessageWithReply
from syft.core.common.message import SignedImmediateSyftMessageWithoutReply

# third party
from nacl.encoding import HexEncoder
from nacl.signing import SigningKey

from .node import node
from .exceptions import (
    PyGridError,
    UserNotFoundError,
    RoleNotFoundError,
    GroupNotFoundError,
    AuthorizationError,
    MissingRequestKeyError,
    InvalidCredentialsError,
)

executor = Executor()


def process_as_syft_message(message_class, message_content, sign_key):
    message = message_class(**message_content)
    signed_message = message.sign(signing_key=sign_key)

    response = {}
    if isinstance(signed_message, SignedImmediateSyftMessageWithReply):
        response = node.recv_immediate_msg_with_reply(
            msg=signed_message, raise_exception=True
        )
        response = response.message
    elif isinstance(signed_message, SignedImmediateSyftMessageWithoutReply):
        node.recv_immediate_msg_without_reply(msg=signed_message)
    else:
        node.recv_eventual_msg_without_reply(msg=signed_message)

    return response


def task_handler(route_function, data, mandatory, optional=[]):
    args = {}
    response_body = {}

    if not data:
        data = {}

    # Fill mandatory args
    for (arg, error) in mandatory.items():
        value = data.get(arg)

        # If not found
        if not value:
            raise error  # Specific Error
        else:
            args[arg] = value  # Add in args dict

    for opt in optional:
        value = data.get(opt)

        # If found
        if value:
            args[opt] = value  # Add in args dict

    # Execute task
    response_body = route_function(**args)

    return response_body


def route_logic(message_class, current_user, msg_content):
    if current_user:
        user_key = SigningKey(
            current_user.private_key.encode("utf-8"), encoder=HexEncoder
        )
        msg_content["internal_key"] = current_user.private_key
        msg_content["current_user"] = current_user.id
    else:
        user_key = SigningKey.generate()

    content = {
        "address": node.address,
        "content": msg_content,
        "reply_to": node.address,
    }

    syft_message = {}
    syft_message["message_class"] = message_class
    syft_message["message_content"] = content
    syft_message["sign_key"] = user_key

    # Execute task
    response_msg = task_handler(
        route_function=process_as_syft_message,
        data=syft_message,
        mandatory={
            "message_class": MissingRequestKeyError,
            "message_content": MissingRequestKeyError,
            "sign_key": MissingRequestKeyError,
        },
    )
    return response_msg
