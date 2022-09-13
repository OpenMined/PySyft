# stdlib
from typing import Generator

# third party
from fastapi import Depends
from fastapi import HTTPException
from fastapi import status
from fastapi.security import OAuth2PasswordBearer
from jose import jwt

# grid absolute
from grid.api.token import TokenPayload
from grid.api.users.models import UserPrivate
from grid.core import security
from grid.core.config import settings
from grid.core.node import node
from grid.db.session import get_db_session

reusable_oauth2 = OAuth2PasswordBearer(tokenUrl=f"{settings.API_V1_STR}/login")


def get_db() -> Generator:
    try:
        db = get_db_session()
        yield db
    finally:
        db.close()


def get_current_user(token: str = Depends(reusable_oauth2)) -> UserPrivate:
    try:
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[security.ALGORITHM]
        )
        token_data = TokenPayload(**payload)
        # TODO: Fix jwt.
        # TODO: Send a secure message with the token instead of fetching the user
        #       directly through node
        user = node.users.first(id_int=int(token_data.sub))
        user_dict = user.to_dict()
        user_dict["id"] = user.id_int
        user_dict["role"] = user.role["name"]
        user_private = UserPrivate(**user_dict)

        return user_private
    except Exception:
        # TODO: Improve error handling
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )


# TODO: Create a dependency for checking permissions
