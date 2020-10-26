from .codes import RESPONSE_MSG
from json.decoder import JSONDecodeError

from .exceptions import (
    PyGridError,
    UserNotFoundError,
    RoleNotFoundError,
    GroupNotFoundError,
    AuthorizationError,
    MissingRequestKeyError,
    InvalidCredentialsError,
)


def task_handler(route_function, data, mandatory, optional=[]):
    args = {}
    response_body = {}

    if not data:
        data = {}

    try:
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
        status_code = 200
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
