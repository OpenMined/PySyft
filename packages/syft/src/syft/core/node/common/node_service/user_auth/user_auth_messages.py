# future
from __future__ import annotations

# stdlib
from datetime import datetime
from datetime import timedelta
from typing import Any
from typing import Dict
from typing import Optional
from typing import Union

# third party
import jwt
from nacl.signing import VerifyKey
from pydantic import BaseSettings
from typing_extensions import final

# relative
from ......logger import error
from .....common.serde.serializable import serializable
from ....abstract.node_service_interface import NodeServiceInterface
from ...exceptions import InvalidCredentialsError
from ..generic_payload.messages import GenericPayloadMessage
from ..generic_payload.messages import GenericPayloadMessageWithReply
from ..generic_payload.messages import GenericPayloadReplyMessage


@serializable(recursive_serde=True)
@final
class UserLoginMessage(GenericPayloadMessage):
    ...


@serializable(recursive_serde=True)
@final
class UserLoginReplyMessage(GenericPayloadReplyMessage):
    ...


# TODO: Move create_access_token from grid to syft
def create_access_token(
    settings: BaseSettings,
    subject: Union[str, Any],
    expires_delta: Optional[timedelta] = None,
    ALGORITHM: str = "HS256",
) -> str:
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
        )
    to_encode = {"exp": expire, "sub": str(subject)}
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


@serializable(recursive_serde=True)
@final
class UserLoginMessageWithReply(GenericPayloadMessageWithReply):
    message_type = UserLoginMessage
    message_reply_type = UserLoginReplyMessage

    def run(
        self, node: NodeServiceInterface, verify_key: Optional[VerifyKey] = None
    ) -> Dict[str, Any]:
        try:
            email = str(self.kwargs["email"])
            password = str(self.kwargs["password"])
            response: Dict[str, Any] = {}
            response["status"] = "error"
            try:
                node.users.login(email=email, password=password)  # type: ignore
            except InvalidCredentialsError as e:
                error(f"Invalid credentials during login for {email}. {e}")
                response["reason"] = "Incorrect email or password"
                return response

            user = node.users.first(email=email)  # type: ignore

            # settings = node.settings
            # access_token_expires = timedelta(
            #     minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
            # )
            # access_token = create_access_token(
            #     settings=settings, subject=user.id, expires_delta=access_token_expires
            # )
            # metadata = (
            #     serialize(node.get_metadata_for_client())
            #     .SerializeToString()
            #     .decode("ISO-8859-1")
            # )

            data = {
                # "access_token": access_token,
                # "token_type": "bearer",
                # "metadata": metadata,
                "key": user.private_key,
            }

            response["status"] = "ok"
            response["data"] = data

            return response
        except Exception as e:
            error(f"Failed to login with {email}. {e}")
            response["status"] = "error"
            return response
