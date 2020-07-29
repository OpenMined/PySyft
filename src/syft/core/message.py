from ..interfaces.message import AbstractMessage
from syft.common.id import UID
from syft.core.io.address import Address


class SyftMessage(AbstractMessage):
    def __init__(self, address: Address, msg_id: UID = None) -> None:
        self.msg_id = msg_id
        self.address = address


# TODO: I think this is really cool but we don't have the notion of a signature yet.
# TODO: (cont) so commenting out for now but will likely bring back in the future
# class SignedMessage(SyftMessage):
#     def sign(self, signature):
#         self.my_route = my_route


class ImmediateSyftMessage(SyftMessage):
    ""


class EventualMessage(SyftMessage):
    ""


class SyftMessageWithReply(SyftMessage):
    ""


class SyftMessageWithoutReply(SyftMessage):
    ""


class ImmediateSyftMessageWithoutReply(ImmediateSyftMessage, SyftMessageWithoutReply):
    def __init__(self, address: Address, msg_id: UID = None) -> None:
        super().__init__(address=address, msg_id=msg_id)


class EventualSyftMessageWithoutReply(EventualMessage, SyftMessageWithoutReply):
    def __init__(self, address: Address, msg_id: UID = None) -> None:
        super().__init__(address=address, msg_id=msg_id)


class ImmediateSyftMessageWithReply(ImmediateSyftMessage, SyftMessageWithReply):
    def __init__(self, reply_to: Address, address: Address, msg_id: UID = None) -> None:
        super().__init__(address=address, msg_id=msg_id)
        self.reply_to = reply_to
