# stdlib
from datetime import timedelta
from typing import Any

# third party
from fastapi import APIRouter
from fastapi import Body
from fastapi import Depends
from fastapi import HTTPException
from fastapi.responses import JSONResponse

# syft absolute
from syft import serialize  # type: ignore
from syft.core.node.common.exceptions import InvalidCredentialsError
from syft.core.node.common.node import Node

# grid absolute
from app import crud
from app import schemas
from app.api import deps
from app.core import security
from app.core.config import settings
from app.core.node import node
from app.core.security import get_password_hash
from app.utils import generate_password_reset_token
from app.utils import send_reset_password_email
from app.utils import verify_password_reset_token

router = APIRouter()


@router.post("/login", name="login", status_code=200, response_class=JSONResponse)
def login_access_token(
    email: str = Body(..., example="info@openmined.org"),
    password: str = Body(..., example="changethis"),
) -> Any:
    """
    You must pass valid credentials to log in. An account in any of the network
    domains is sufficient for logging in.
    """
    try:
        node.users.login(email=email, password=password)
    except InvalidCredentialsError:
        raise HTTPException(status_code=401, detail="Incorrect email or password")

    user = node.users.first(email=email)

    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = security.create_access_token(
        user.id, expires_delta=access_token_expires
    )
    metadata = (
        serialize(node.get_metadata_for_client())
        .SerializeToString()
        .decode("ISO-8859-1")
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "metadata": metadata,
        "key": user.private_key,
    }


@router.post("/login/test-token", response_model=schemas.User)
def test_token(current_user: Any = Depends(deps.get_current_user)) -> Any:
    """
    Test access token
    """
    return current_user


@router.post("/password-recovery/{email}", response_model=schemas.Msg)
def recover_password(email: str, node: Node = Depends(deps.get_db)) -> Any:
    """
    Password Recovery
    """
    db = node.db
    user = crud.user.get_by_email(db, email=email)

    if not user:
        raise HTTPException(
            status_code=404,
            detail="The user with this username does not exist in the system.",
        )
    password_reset_token = generate_password_reset_token(email=email)
    send_reset_password_email(
        email_to=user.email, email=email, token=password_reset_token
    )
    return {"msg": "Password recovery email sent"}


@router.post("/reset-password/", response_model=schemas.Msg)
def reset_password(
    token: str = Body(...),
    new_password: str = Body(...),
    node: Node = Depends(deps.get_db),
) -> Any:
    """
    Reset password
    """
    db = node.db
    email = verify_password_reset_token(token)
    if not email:
        raise HTTPException(status_code=400, detail="Invalid token")
    user = crud.user.get_by_email(db, email=email)
    if not user:
        raise HTTPException(
            status_code=404,
            detail="The user with this username does not exist in the system.",
        )
    elif not crud.user.is_active(user):
        raise HTTPException(status_code=400, detail="Inactive user")
    hashed_password = get_password_hash(new_password)
    user.hashed_password = hashed_password
    db.add(user)
    db.commit()
    return {"msg": "Password updated successfully"}
