from syft_rpc import JSONModel
from syft_rpc import Future

OBJECT_TYPE_HEADER = "x-syft-rpc-object-type"


class User(JSONModel):
    id: int
    name: str


class LoginResponse(JSONModel):
    username: str
    token: int = 123


TypeRegistry = {"User": User, "LoginResponse": LoginResponse}


def to_obj(obj, headers):
    if OBJECT_TYPE_HEADER in headers and headers[OBJECT_TYPE_HEADER] in TypeRegistry:
        constructor = TypeRegistry[headers[OBJECT_TYPE_HEADER]]
        return constructor(**obj)


def body_to_obj(message):
    if isinstance(message, Future):
        message = message.wait()
    return to_obj(message.decode(), message.headers)
