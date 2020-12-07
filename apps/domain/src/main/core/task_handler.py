from .codes import RESPONSE_MSG
from json.decoder import JSONDecodeError
from flask_executor import Executor


from syft.core.common.message import SignedImmediateSyftMessageWithReply
from syft.core.common.message import SignedImmediateSyftMessageWithoutReply

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
    message = message_class(address=node.address)
    signed_message = message.sign(signing_key=sign_key)
    print("My Message: ", message)
    print("My Signed Message: ", signed_message)

    response_dict = {}
    if isinstance(signed_message, SignedImmediateSyftMessageWithReply):
        response_dict = node.recv_immediate_msg_with_reply(msg=obj_msg)
    elif isinstance(signed_message, SignedImmediateSyftMessageWithoutReply):
        node.recv_immediate_msg_without_reply(msg=signed_message)
    else:
        node.recv_eventual_msg_without_reply(msg=signed_message)

    return response_dict


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
