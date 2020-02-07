import json
from ..scopes import scopes
from ..codes import GRID_MSG
from .socket_handler import SocketHandler

handler = SocketHandler()


def scope_broadcast(message: dict, socket) -> str:
    """ It will perform a broadcast to all nodes registered at the same scope that was sended on the body of this message.
    
        Args:
            message: Message used to find the specific scope and that will be forwarded (It can be join-room and peer-left)
            socket: Socket descriptor.
        Returns:
            response: An empty response or an error message.
    """
    data = message[GRID_MSG.DATA_FIELD]
    try:
        scope_id = data.get("scopeId", None)
        worker_id = data.get("workerId", None)
        scope = scopes.get_scope(scope_id)
        for worker in scope.assignments.keys():
            if worker != worker_id:
                handler.send_msg(worker, json.dumps(message))
        return ""
    except Exception as e:
        return str(e)


def internal_message(message: dict, socket) -> str:
    """ It will forward an webrtc - internal message to the desired node.
        
        Args:
            message: Message used to find the desired destination and that will be forwarded.
            socket: Socket descriptor.
        Returns:
            response: An empty response or an error message.
    """
    data = message[GRID_MSG.DATA_FIELD]
    try:
        destination = data.get("to", None)
        if destination:
            handler.send_msg(destination, json.dumps(message))
        return ""
    except Exception as e:
        return str(e)
