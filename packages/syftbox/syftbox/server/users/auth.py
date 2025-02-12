from datetime import datetime, timezone
from typing import Any

import jwt
from fastapi import Depends, HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from opentelemetry import trace
from typing_extensions import Annotated

from syftbox.server.settings import ServerSettings, get_server_settings
from syftbox.server.telemetry import OTEL_ATTR_CLIENT_USER

bearer_scheme = HTTPBearer()

ACCESS_TOKEN = "access_token"
EMAIL_TOKEN = "email_token"


def _validate_jwt(server_settings: ServerSettings, token: str) -> dict:
    try:
        return jwt.decode(
            token,
            server_settings.jwt_secret.get_secret_value(),
            algorithms=[server_settings.jwt_algorithm],
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=401,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception:
        raise HTTPException(
            status_code=401,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )


def _generate_jwt(server_settings: ServerSettings, data: dict) -> str:
    return jwt.encode(
        data,
        server_settings.jwt_secret.get_secret_value(),
        algorithm=server_settings.jwt_algorithm,
    )


def generate_access_token(server_settings: ServerSettings, email: str) -> str:
    iat: datetime = datetime.now(tz=timezone.utc)
    data: dict[str, Any] = {
        "email": email,
        "type": ACCESS_TOKEN,
        "iat": iat,
    }
    if server_settings.jwt_access_token_exp:
        data["exp"] = iat + server_settings.jwt_access_token_exp
    return _generate_jwt(server_settings, data)


def generate_email_token(server_settings: ServerSettings, email: str) -> str:
    iat: datetime = datetime.now(tz=timezone.utc)
    data: dict[str, Any] = {
        "email": email,
        "type": EMAIL_TOKEN,
        "iat": iat,
    }
    if server_settings.jwt_email_token_exp:
        data["exp"] = iat + server_settings.jwt_email_token_exp
    return _generate_jwt(server_settings, data)


def validate_access_token(server_settings: ServerSettings, token: str) -> dict:
    data = _validate_jwt(server_settings, token)
    if data["type"] != ACCESS_TOKEN:
        raise HTTPException(
            status_code=401,
            detail="Invalid token type",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return data


def validate_email_token(server_settings: ServerSettings, token: str) -> dict:
    data = _validate_jwt(server_settings, token)
    if data["type"] != EMAIL_TOKEN:
        raise HTTPException(
            status_code=401,
            detail="Invalid token type",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return data


def get_user_from_email_token(
    credentials: Annotated[HTTPAuthorizationCredentials, Security(bearer_scheme)],
    server_settings: Annotated[ServerSettings, Depends(get_server_settings)],
) -> str:
    payload = validate_email_token(server_settings, credentials.credentials)
    trace.get_current_span().set_attribute(OTEL_ATTR_CLIENT_USER, payload["email"])
    return payload["email"]


def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials, Security(bearer_scheme)],
    server_settings: Annotated[ServerSettings, Depends(get_server_settings)],
) -> str:
    payload = validate_access_token(server_settings, credentials.credentials)
    trace.get_current_span().set_attribute(OTEL_ATTR_CLIENT_USER, payload["email"])
    return payload["email"]
