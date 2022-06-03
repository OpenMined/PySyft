# stdlib
import logging
from typing import Any
from typing import Dict
from typing import Optional
from typing import Type

# relative
from .....grid import GridURL
from .....lib.python.util import upcast
from ...abstract.node import AbstractNodeClient
from ..action.exception_action import ExceptionMessage
from ..node_service.generic_payload.messages import GenericPayloadMessageWithReply
from ..node_service.vpn.vpn_messages import VPNJoinMessageWithReply
from ..node_service.vpn.vpn_messages import VPNStatusMessageWithReply


class VPNAPI:
    def __init__(self, client: AbstractNodeClient):
        self.client = client

    def join_network_vpn(self, grid_url: GridURL) -> None:
        reply = self.perform_api_request(
            syft_msg=VPNJoinMessageWithReply, content={"grid_url": grid_url}
        )
        logging.info(reply.payload)
        status = "error"
        try:
            status = str(reply.payload.kwargs.get("status"))
        except Exception:  # nosec
            pass

        if status == "ok":
            # print(f"🔌 {self.client} successfully connected to the VPN: {grid_url}")
            pass
        else:
            print(f"❌ {self.client} failed to connect to the VPN: {grid_url}")

    def get_status(self) -> Dict[str, Any]:
        reply = self.perform_api_request(syft_msg=VPNStatusMessageWithReply, content={})
        logging.info(reply.payload)
        try:
            return upcast(reply.payload.kwargs)
        except Exception:  # nosec
            pass

        print(f"❌ {self.client} VPN Status failed")
        return {"status": "error"}

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
