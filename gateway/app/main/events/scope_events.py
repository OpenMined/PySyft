from ..codes import GRID_MSG, RESPONSE_MSG
from ..scopes import scopes
from .socket_handler import SocketHandler
import uuid
import json

# Singleton socket handler
handler = SocketHandler()


def get_protocol(message: dict, socket) -> str:
    """ This endpoint is used to create a new scope or add a new participant
        in a previously registered scope.

        Args:
            message : Message body sended by some client.
            socket: Socket descriptor.
        Returns:
            response : String response to the client
    """
    data = message[GRID_MSG.DATA_FIELD]

    # If something goes wrong it will return the exception message.
    try:
        worker_id = data.get("workerId", None)
        scope_id = data.get("scopeId", None)
        protocol_id = data.get("protocolId", None)

        # If worker id was not sended, we need to create a new one.
        if not worker_id:
            # Create a new worker ID
            # Attach this ID on this ws connection.
            worker_id = str(uuid.uuid4())

        # Create a link between worker id and socket descriptor
        handler.new_connection(worker_id, socket)

        # If scope id wasn't sended, we need to create a new scope
        if not scope_id:
            # Create new scope.
            scope = scopes.create_scope(worker_id, protocol_id)
            scope_id = scope.id
        else:  # Try to find the desired scope.
            scope = scopes.get_scope(scope_id)

        # Add the new participant
        scope.add_participant(worker_id)

        # Build response
        data = {}

        data["user"] = {
            RESPONSE_MSG.WORKER_ID: worker_id,
            RESPONSE_MSG.SCOPE_ID: scope_id,
            RESPONSE_MSG.PROTOCOL_ID: scope.protocol,
            RESPONSE_MSG.ROLE: scope.get_role(worker_id),
            RESPONSE_MSG.PLAN: scope.get_plan(worker_id),
            RESPONSE_MSG.ASSIGNMENT: scope.get_assignment(worker_id),
        }

        data["participants"] = {
            p: scope.get_assignment(p) for p in scope.get_participants()
        }

        data = json.dumps(data)
        response = {"type": GRID_MSG.GET_PROTOCOL, "data": data}

        return json.dumps(response)
    except Exception as e:
        return str(e)
