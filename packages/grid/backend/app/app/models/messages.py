# stdlib
from typing import Any
from typing import Type
from typing import Union

# third party
from nacl.signing import SigningKey
from pydantic import BaseModel

# syft absolute
# syft
from syft.core.common.message import ImmediateSyftMessageWithReply
from syft.core.common.message import SignedImmediateSyftMessageWithReply
from syft.core.common.message import SignedImmediateSyftMessageWithoutReply
from syft.core.node.common.action.exception_action import ExceptionMessage

# grid absolute
from app.core.node import node


# TODO: Add without reply?
# TODO: Check if its necessary to break into multiple messages (Signed, Unsigned, etc)
class GridMessage(BaseModel):
    signing_key: SigningKey
    with_reply: bool = True
    is_signed: bool = True
    syft_message_type: Union[
        Type[ImmediateSyftMessageWithReply], Type[SignedImmediateSyftMessageWithReply]
    ]
    _message: SignedImmediateSyftMessageWithReply
    _response: Any = None

    class Config:
        arbitrary_types_allowed = True
        underscore_attrs_are_private = True
        orm_mode = True

    def create_message(self, **kwargs: Any) -> None:
        self._message = self.syft_message_type(
            address=node.address, reply_to=node.address, **kwargs
        )
        if self.is_signed:
            self.sign_message()

    def submit_message_with_reply(self) -> None:
        self._response = node.recv_immediate_msg_with_reply(msg=self._message).message

    def sign_message(self) -> None:
        self._message = self._message.sign(self.signing_key)

    def validate_syft_response(self) -> None:
        if self._response is not None and isinstance(self._response, ExceptionMessage):
            raise Exception(self._response.exception_msg)

    def send_with_reply(
        self, **kwargs: Any
    ) -> SignedImmediateSyftMessageWithoutReply:  # TODO: Improve this type
        self.create_message(**kwargs)
        self.submit_message_with_reply()
        self.validate_syft_response()
        return self._response
