# stdlib
from datetime import timedelta
from typing import Any

# third party
from fastapi import APIRouter
from fastapi import Body
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from loguru import logger
from nacl.encoding import HexEncoder
from nacl.signing import SigningKey

# syft absolute
from syft import serialize  # type: ignore
from syft.core.common.uid import UID
from syft.core.node.common.exceptions import InvalidCredentialsError
from syft.core.node.new.credentials import SyftVerifyKey

# grid absolute
from grid.core import security
from grid.core.config import settings
from grid.core.node import node
from grid.core.node import worker

router = APIRouter()


@router.post("/key", name="key", status_code=200, response_class=JSONResponse)
def auth_using_signing_key(
    signing_key: str = Body(..., embed=True),
) -> Any:
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    user = node.users.first(private_key=signing_key)

    access_token = security.create_access_token(
        user.id,
        expires_delta=access_token_expires,
    )

    metadata = serialize(node.get_metadata_for_client(), to_bytes=True).decode(
        "ISO-8859-1"
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "metadata": metadata,
    }


@router.post("/guest", name="guest", status_code=200, response_class=JSONResponse)
def guest_user() -> Any:
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    active_guests = len(node.guest_signing_key_registry)
    access_token = security.create_access_token(
        active_guests + 1,
        expires_delta=access_token_expires,
        guest=True,
    )
    metadata = serialize(node.get_metadata_for_client(), to_bytes=True).decode(
        "ISO-8859-1"
    )
    guest_signing_key = (
        SigningKey.generate()
    )  # .encode(encoder=HexEncoder).decode("utf-8")
    node.guest_signing_key_registry.add(guest_signing_key)
    node.guest_verify_key_registry.add(guest_signing_key.verify_key)

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "metadata": metadata,
        "key": guest_signing_key.encode(encoder=HexEncoder).decode("utf-8"),
    }


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


@router.post(
    "/new_login", name="new_login", status_code=200, response_class=JSONResponse
)
def new_login(
    email: str = Body(..., example="info@openmined.org"),
    password: str = Body(..., example="changethis"),
) -> Any:
    """
    You must pass valid credentials to log in.
    """

    method = worker._get_service_method_from_path("UserCollection.verify")
    result = method(credentials=worker.signing_key, email=email, password=password)
    if result.is_err():
        logger.bind(payload={"email": email}).error(result.err())
        return {"Error": result.err()}

    user = result.ok()

    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = security.create_access_token(
        user.id, expires_delta=access_token_expires
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "node_name": worker.name,
        "node_uid": worker.id.no_dash,
        "user_key": str(user.verify_key),
        "user_id": str(user.id.no_dash),
    }
