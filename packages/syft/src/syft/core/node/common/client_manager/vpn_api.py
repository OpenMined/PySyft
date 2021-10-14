# stdlib
import logging
from typing import Any
from typing import Dict
from typing import Optional
from typing import Type

# relative
from ...abstract.node import AbstractNodeClient
from ..action.exception_action import ExceptionMessage
from ..node_service.generic_payload.messages import GenericPayloadMessageWithReply
from ..node_service.vpn.vpn_messages import VPNJoinMessageWithReply


class VPNAPI:
    def __init__(self, client: AbstractNodeClient):
        self.client = client

    def join_network(self, host_or_ip: str) -> None:
        response = self.perform_api_request(
            syft_msg=VPNJoinMessageWithReply, content={"host_or_ip": host_or_ip}
        )
        logging.info(response.resp_msg)

    def perform_api_request(
        self,
        syft_msg: Optional[Type[GenericPayloadMessageWithReply]],
        content: Optional[Dict[Any, Any]] = None,
    ) -> Any:
        if syft_msg is None:
            raise ValueError(
                "Can't perform this type of api request, the message is None."
            )
        else:
            syft_msg_constructor = syft_msg

        if content is None:
            content = {}
        signed_msg = (
            syft_msg_constructor(kwargs=content)
            .to(address=self.client.address, reply_to=self.client.address)
            .sign(signing_key=self.client.signing_key)
        )  # type: ignore
        response = self.client.send_immediate_msg_with_reply(msg=signed_msg)
        if isinstance(response, ExceptionMessage):
            raise response.exception_type
        else:
            return response
