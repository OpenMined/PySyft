import json

from .. import ws
from ..codes import GRID_EVENTS, USER_EVENTS, ROLE_EVENTS
from .network import *
from .user_related import *
from .role_related import *
from .socket_handler import SocketHandler

socket_handler = SocketHandler()

routes = {
    GRID_EVENTS.JOIN: register_node,
    GRID_EVENTS.MONITOR_ANSWER: update_node,
    GRID_EVENTS.FORWARD: forward,
    ROLE_EVENTS.CREATE_ROLE: create_role_socket,
    ROLE_EVENTS.GET_ALL_ROLES: get_all_roles_socket,
    ROLE_EVENTS.GET_SPECIFIC_ROLE: get_role_socket,
    ROLE_EVENTS.PUT_ROLE: put_role_socket,
    ROLE_EVENTS.DELETE_ROLE: delete_role_socket,
    USER_EVENTS.SIGNUP_USER: signup_user_socket,
    USER_EVENTS.LOGIN_USER: login_user_socket,
    USER_EVENTS.GET_ALL_USERS: get_all_users_socket,
    USER_EVENTS.GET_SPECIFIC_USER: get_specific_user_socket,
    USER_EVENTS.SEARCH_USERS: search_users_socket,
    USER_EVENTS.UPDATE_USER_EMAIL: change_user_email_socket,
    USER_EVENTS.UPDATE_USER_PASSWORD: change_user_password_socket,
    USER_EVENTS.UPDATE_USER_ROLE: change_user_role_socket,
    USER_EVENTS.DELETE_USER: delete_user_socket,
}


def route_request(message, socket):
    global routes

    message = json.loads(message)
    print("Message: ", message)
    if message and message.get(MSG_FIELD.TYPE, None) in routes.keys():
        return routes[message[MSG_FIELD.TYPE]](message, socket)
    else:
        return {"status": "error", "message": "Invalid request format!"}


@ws.route("/")
def socket_api(socket):
    while not socket.closed:
        message = socket.receive()
        if not message:
            continue
        else:
            response = route_request(message, socket)
            if response:
                socket.send(json.dumps(response))

    worker_id = socket_handler.remove(socket)
