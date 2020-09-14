from functools import wraps
from json import dumps

from json.decoder import JSONDecodeError
from flask import current_app as app
from syft.codes import RESPONSE_MSG
import jwt

from .core.exceptions import (
    PyGridError,
    UserNotFoundError,
    RoleNotFoundError,
    GroupNotFoundError,
    AuthorizationError,
    MissingRequestKeyError,
    InvalidCredentialsError,
)
from .database import User
from .. import db


def token_required_factory(get_token, format_result):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            status_code = 200
            mimetype = "application/json"
            response_body = {}
            try:
                token = get_token(*args, **kwargs)
                if token is None:
                    raise MissingRequestKeyError
            except Exception as e:
                status_code = 400  # Bad Request
                response_body[RESPONSE_MSG.ERROR] = str(e)
                return format_result(response_body, status_code, mimetype)

            try:
                data = jwt.decode(token, app.config["SECRET_KEY"], algorithms="HS256")
                current_user = User.query.get(data["id"])
                if current_user is None:
                    raise UserNotFoundError
            except Exception as e:
                status_code = 403  # Unauthorized
                response_body[RESPONSE_MSG.ERROR] = str(InvalidCredentialsError())
                return format_result(response_body, status_code, mimetype)

            return f(current_user, *args, **kwargs)

        return wrapper

    return decorator


def error_handler(f, *args, **kwargs):
    status_code = 200  # Success
    response_body = {}

    try:
        response_body = f(*args, **kwargs)

    except (InvalidCredentialsError, AuthorizationError) as e:
        status_code = 403  # Unathorized
        response_body[RESPONSE_MSG.ERROR] = str(e)
    except (GroupNotFoundError, RoleNotFoundError, UserNotFoundError) as e:
        status_code = 404  # Resource not found
        response_body[RESPONSE_MSG.ERROR] = str(e)
    except (TypeError, MissingRequestKeyError, PyGridError, JSONDecodeError) as e:
        status_code = 400  # Bad Request
        response_body[RESPONSE_MSG.ERROR] = str(e)
    except Exception as e:
        status_code = 500  # Internal Server Error
        response_body[RESPONSE_MSG.ERROR] = str(e)

    return status_code, response_body
