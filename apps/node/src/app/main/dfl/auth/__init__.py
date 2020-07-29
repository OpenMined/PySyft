# Python standard imports
import functools
import json

# External imports
from flask_login import LoginManager, UserMixin, current_user

# Local imports
from ... import local_worker
from .session_repository import SessionsRepository
from .user_session import UserSession

SESSION_TYPES = [UserSession]
session_repository = None
login_manager = LoginManager()


def set_auth_configs(app):
    """Set configs to use flask session manager.

    Args:
        app: Flask application
    Returns:
        app: Flask application
    """
    global session_repository
    login_manager.init_app(app)
    session_repository = SessionsRepository()
    return app


def get_session():
    """Returns the global instance of session repository."""
    global session_repository
    return session_repository


# callback to reload the user object
@login_manager.user_loader
def load_user(user_id: str) -> UserMixin:
    """Retrieve user session object from session repository.

    Args:
        user_id (str) : User id.
    Returns:
        user : User Session.
    """
    return session_repository.get_session_by_id(user_id)


def authenticated_only(f):
    """Custom Wrapper to check and route authenticated user sessions.

    Args:
        f (function) : Function to be used by authenticated users.
    Returns:
        response : Function result.
    """

    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        if not current_user.is_authenticated:
            current_user.worker = local_worker
            return f(*args, **kwargs)
        else:
            return f(*args, **kwargs)

    return wrapped
