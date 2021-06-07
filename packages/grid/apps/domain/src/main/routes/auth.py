# stdlib
from functools import wraps
from json import dumps
from json.decoder import JSONDecodeError

# third party
from flask import Response
from flask import current_app as app
from flask import request
import jwt
from syft.core.node.common.node import DuplicateRequestException

# grid relative
from ..core.codes import RESPONSE_MSG
from ..core.database import User
from ..core.database import db
from ..core.exceptions import AuthorizationError
from ..core.exceptions import GroupNotFoundError
from ..core.exceptions import InvalidCredentialsError
from ..core.exceptions import MissingRequestKeyError
from ..core.exceptions import OwnerAlreadyExistsError
from ..core.exceptions import PyGridError
from ..core.exceptions import RoleNotFoundError
from ..core.exceptions import UserNotFoundError


def token_required_factory(get_token, format_result, optional=False):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            status_code = 200
            mimetype = "application/json"
            response_body = {}
            try:
                token = get_token(optional=optional)
            except Exception as e:
                status_code = 400  # Bad Request
                response_body[RESPONSE_MSG.ERROR] = str(e)
                return format_result(response_body, status_code, mimetype)
            try:
                current_user = None
                if token:
                    data = jwt.decode(
                        token, app.config["SECRET_KEY"], algorithms="HS256"
                    )
                    current_user = User.query.get(data["id"])
                if current_user is None and not optional:
                    raise UserNotFoundError
            except Exception as e:
                status_code = 403  # Unauthorized
                response_body[RESPONSE_MSG.ERROR] = str(InvalidCredentialsError())
                return format_result(response_body, status_code, mimetype)

            return f(current_user, *args, **kwargs)

        return wrapper

    return decorator


def get_token(optional=False):
    token = request.headers.get("token", None)
    if token is None and not optional:
        raise MissingRequestKeyError

    return token


def format_result(response_body, status_code, mimetype):
    return Response(dumps(response_body), status=status_code, mimetype=mimetype)


token_required = token_required_factory(get_token, format_result)
optional_token = token_required_factory(get_token, format_result, optional=True)


def error_handler(f, success_code, *args, **kwargs):
    status_code = success_code  # Success
    response_body = {}

    try:
        response_body = f(*args, **kwargs)
    except (
        InvalidCredentialsError,
        AuthorizationError,
        DuplicateRequestException,
    ) as e:
        status_code = 403  # Unathorized
        response_body[RESPONSE_MSG.ERROR] = str(e)
    except (GroupNotFoundError, RoleNotFoundError, UserNotFoundError) as e:
        status_code = 404  # Resource not found
        response_body[RESPONSE_MSG.ERROR] = str(e)
    except (OwnerAlreadyExistsError) as e:
        status_code = 409  # Conflict!
        response_body[RESPONSE_MSG.ERROR] = str(e)
    except (TypeError, MissingRequestKeyError, PyGridError, JSONDecodeError) as e:
        status_code = 400  # Bad Request
        response_body[RESPONSE_MSG.ERROR] = str(e)
    except Exception as e:
        status_code = 500  # Internal Server Error
        response_body[RESPONSE_MSG.ERROR] = str(e)

    return status_code, response_body
