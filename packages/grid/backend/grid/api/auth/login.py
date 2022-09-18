# stdlib
from datetime import timedelta
from typing import Any

# third party
from fastapi import APIRouter
from fastapi import Body
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from loguru import logger

# syft absolute
from syft import serialize  # type: ignore
from syft.core.node.common.exceptions import InvalidCredentialsError

# grid absolute
from grid.core import security
from grid.core.config import settings
from grid.core.node import node

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
        user = node.users.login(email=email, password=password)
    except InvalidCredentialsError as err:
        logger.bind(payload={"email": email}).error(err)
        raise HTTPException(status_code=401, detail="Incorrect email or password")

    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = security.create_access_token(
        user.id_int, expires_delta=access_token_expires
    )
    metadata = serialize(node.get_metadata_for_client(), to_bytes=True).decode(
        "ISO-8859-1"
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "metadata": metadata,
        "key": user.private_key,
    }
