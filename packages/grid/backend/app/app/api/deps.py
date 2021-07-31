# stdlib
from typing import Generator

# third party
from fastapi import Depends
from fastapi import HTTPException
from fastapi import status
from fastapi.security import OAuth2PasswordBearer
from jose import jwt

# grid absolute
from app import schemas
from app.core import security
from app.core.config import settings
from app.core.node import node
from app.db.session import SessionLocal
from app.users.models import UserPrivate

reusable_oauth2 = OAuth2PasswordBearer(tokenUrl=f"{settings.API_V1_STR}/login")


def get_db() -> Generator:
    try:
        db = SessionLocal()
        yield db
    finally:
        db.close()


def get_current_user(token: str = Depends(reusable_oauth2)) -> UserPrivate:
    try:
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[security.ALGORITHM]
        )
        token_data = schemas.TokenPayload(**payload)
        # TODO: Fix jwt.
        # TODO: Send a secure message with the token instead of fetching the user
        #       directly through node
        user = node.users.first(id=token_data.sub)
        current_user = UserPrivate.from_orm(user)
        return current_user
    except Exception:
        # TODO: Improve error handling
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )


# TODO: Create a dependency for checking permissions
