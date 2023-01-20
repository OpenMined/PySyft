# stdlib
from typing import List
from typing import Optional
from typing import Type

# third party
from nacl.signing import VerifyKey

# relative
from ......util import traceback_and_raise
from ....abstract.node_service_interface import NodeServiceInterface
from ..auth import service_auth
from ..node_service import ImmediateNodeServiceWithReply
from .vpn_messages import VPNConnectMessage
from .vpn_messages import VPNConnectMessageWithReply
from .vpn_messages import VPNConnectReplyMessage
from .vpn_messages import VPNJoinMessage
from .vpn_messages import VPNJoinMessageWithReply
from .vpn_messages import VPNJoinReplyMessage
from .vpn_messages import VPNJoinSelfMessage
from .vpn_messages import VPNJoinSelfMessageWithReply
from .vpn_messages import VPNJoinSelfReplyMessage
from .vpn_messages import VPNRegisterMessage
from .vpn_messages import VPNRegisterMessageWithReply
from .vpn_messages import VPNRegisterReplyMessage
from .vpn_messages import VPNStatusMessage
from .vpn_messages import VPNStatusMessageWithReply
from .vpn_messages import VPNStatusReplyMessage


class VPNConnectService(ImmediateNodeServiceWithReply):
    @staticmethod
    @service_auth(guests_welcome=True)
    def process(
        node: NodeServiceInterface,
        msg: VPNConnectMessage,
        verify_key: Optional[VerifyKey] = None,
    ) -> VPNConnectReplyMessage:
        if verify_key is None:
            traceback_and_raise(
                "Can't process VPNConnectService with no verification key."
            )

        result = msg.payload.run(node=node, verify_key=verify_key)
        return VPNConnectMessageWithReply(kwargs=result).back_to(address=msg.reply_to)

    @staticmethod
    def message_handler_types() -> List[Type[VPNConnectMessage]]:
        return [VPNConnectMessage]


class VPNJoinService(ImmediateNodeServiceWithReply):
    @staticmethod
    @service_auth(guests_welcome=True)
    def process(
        node: NodeServiceInterface,
        msg: VPNJoinMessage,
        verify_key: Optional[VerifyKey] = None,
    ) -> VPNJoinReplyMessage:
        if verify_key is None:
            traceback_and_raise(
                "Can't process VPNJoinService with no verification key."
            )

        result = msg.payload.run(node=node, verify_key=verify_key)
        return VPNJoinMessageWithReply(kwargs=result).back_to(address=msg.reply_to)

    @staticmethod
    def message_handler_types() -> List[Type[VPNJoinMessage]]:
        return [VPNJoinMessage]


class VPNJoinSelfService(ImmediateNodeServiceWithReply):
    @staticmethod
    @service_auth(guests_welcome=True)
    def process(
        node: NodeServiceInterface,
        msg: VPNJoinSelfMessage,
        verify_key: Optional[VerifyKey] = None,
    ) -> VPNJoinSelfReplyMessage:
        if verify_key is None:
            traceback_and_raise(
                "Can't process VPNJoinService with no verification key."
            )

        result = msg.payload.run(node=node, verify_key=verify_key)
        return VPNJoinSelfMessageWithReply(kwargs=result).back_to(address=msg.reply_to)

    @staticmethod
    def message_handler_types() -> List[Type[VPNJoinSelfMessage]]:
        return [VPNJoinSelfMessage]


class VPNRegisterService(ImmediateNodeServiceWithReply):
    @staticmethod
    @service_auth(guests_welcome=True)
    def process(
        node: NodeServiceInterface,
        msg: VPNRegisterMessage,
        verify_key: Optional[VerifyKey] = None,
    ) -> VPNRegisterReplyMessage:
        # this service requires no verify_key because its currently public
        result = msg.payload.run(node=node)
        return VPNRegisterMessageWithReply(kwargs=result).back_to(address=msg.reply_to)

    @staticmethod
    def message_handler_types() -> List[Type[VPNRegisterMessage]]:
        return [VPNRegisterMessage]


class VPNStatusService(ImmediateNodeServiceWithReply):
    @staticmethod
    @service_auth(guests_welcome=True)
    def process(
        node: NodeServiceInterface,
        msg: VPNStatusMessage,
        verify_key: Optional[VerifyKey] = None,
    ) -> VPNStatusReplyMessage:
        if verify_key is None:
            traceback_and_raise(
                "Can't process VPNJoinService with no verification key."
            )

        result = msg.payload.run(node=node)
        return VPNStatusMessageWithReply(kwargs=result).back_to(address=msg.reply_to)

    @staticmethod
    def message_handler_types() -> List[Type[VPNStatusMessage]]:
        return [VPNStatusMessage]
