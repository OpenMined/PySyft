# stdlib
import logging

# relative
from ...abstract_server import AbstractServer
from .user import User
from .user import UserCreate
from .user_roles import ServiceRole

logger = logging.getLogger(__name__)


def create_root_admin_if_not_exists(
    name: str,
    email: str,
    password: str,
    server: AbstractServer,
) -> User | None:
    """
    If no root admin exists:
    - all exists checks on the user stash will fail, as we cannot get the role for the admin to check if it exists
    - result: a new admin is always created

    If a root admin exists with a different email:
    - cause: DEFAULT_USER_EMAIL env variable is set to a different email than the root admin in the db
    - verify_key_exists will return True
    - result: no new admin is created, as the server already has a root admin
    """
    user_stash = server.services.user.stash

    email_exists = user_stash.email_exists(email=email).unwrap()
    if email_exists:
        logger.debug("Admin not created, a user with this email already exists")
        return None

    verify_key_exists = user_stash.verify_key_exists(server.verify_key).unwrap()
    if verify_key_exists:
        logger.debug("Admin not created, this server already has a root admin")
        return None

    create_user = UserCreate(
        name=name,
        email=email,
        password=password,
        password_verify=password,
        role=ServiceRole.ADMIN,
    )

    # New User Initialization
    # 🟡 TODO: change later but for now this gives the main user super user automatically
    user = create_user.to(User)
    user.signing_key = server.signing_key
    user.verify_key = server.verify_key

    new_user = user_stash.set(
        credentials=server.verify_key,
        obj=user,
        ignore_duplicates=False,
    ).unwrap()

    logger.debug(f"Created admin {new_user.email}")

    return new_user
