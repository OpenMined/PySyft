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

# grid absolute
from app import schemas
from app.api import deps
from app.core import security
from app.core.config import settings
from app.core.node import node

router = APIRouter()


@router.post("/login", status_code=200, response_class=JSONResponse)
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
