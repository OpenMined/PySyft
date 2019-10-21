import syft as sy
import json
from flask_login import current_user
from .. import local_worker, hook
from ..persistence.utils import recover_objects, snapshot
from ..auth import authenticated_only


@authenticated_only
def forward_binary_message(message: bin) -> bin:
    """ Forward binary syft messages to user's workers.

        Args:
            message (bin) : PySyft binary message.
        Returns:
            response (bin) : PySyft binary response.
    """

    # If worker is empty, load previous database tensors
    if not current_user.worker._objects:
        recover_objects(current_user.worker)

    # Process message
    decoded_response = current_user.worker._recv_msg(message)

    # Save worker state at database
    snapshot(current_user.worker)

    return decoded_response


@authenticated_only
def syft_command(message: dict) -> str:
    """ Forward JSON syft messages to user's workers.

        Args:
            message (dict) : Dictionary data structure containing PySyft message.
        Returns:
            response (str) : Node response.
    """
    response = local_worker._message_router[message["msg_type"]](message["content"])
    payload = sy.serde.serialize(response, force_no_serialization=True)
    return json.dumps({"type": "command-response", "response": payload})
