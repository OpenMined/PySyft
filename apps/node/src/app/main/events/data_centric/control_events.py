# Standard python
import json

# External modules
from flask_login import login_user
from syft.codes import RESPONSE_MSG
from syft.grid.clients.data_centric_fl_client import DataCentricFLClient

from ... import hook, local_worker, sy

# Local imports
from ...core.codes import MSG_FIELD
from ...data_centric.auth import authenticated_only, get_session


def get_node_infos(message: dict) -> str:
    """Returns node id.

    Returns:
        response (str) : Response message containing node id.
    """
    return json.dumps(
        {
            RESPONSE_MSG.NODE_ID: local_worker.id,
            MSG_FIELD.SYFT_VERSION: sy.version.__version__,
        }
    )


def authentication(message: dict) -> str:
    """Receive user credentials and performs user authentication.

    Args:
        message (dict) : Dict data structure containing user credentials.
    Returns:
        response (str) : Authentication response message.
    """
    user = get_session().authenticate(message)
    # If it was authenticated
    if user:
        login_user(user)
        return json.dumps(
            {RESPONSE_MSG.SUCCESS: "True", RESPONSE_MSG.NODE_ID: user.worker.id}
        )
    else:
        return json.dumps({RESPONSE_MSG.ERROR: "Invalid username/password!"})


def connect_grid_nodes(message: dict) -> str:
    """Connect remote grid nodes between each other.

    Args:
        message (dict) :  Dict data structure containing node_id, node address and user credentials(optional).
    Returns:
        response (str) : response message.
    """
    if message["id"] not in local_worker._known_workers:
        worker = DataCentricFLClient(hook, address=message["address"], id=message["id"])
    return json.dumps({"status": "Succesfully connected."})


@authenticated_only
def socket_ping(message: dict) -> str:
    """Ping request to check node's health state."""
    return json.dumps({"alive": "True"})
