# syft absolute
import syft as sy
from syft.core.node.common.node_service.get_repr.get_repr_messages import GetReprMessage
from syft.core.node.common.node_service.ping.ping_messages import PingMessageWithReply
from syft.core.node.common.node_service.simple.simple_messages import (
    NodeRunnableMessageWithReply,
)
from syft.core.node.common.node_service.tff.tff_messages import TFFMessageWithReply
from syft.grid import GridURL

domain = sy.login(email="info@openmined.org", password="changethis", port=8081)

# msg = NodeRunnableMessageWithReply("Hello")
msg = TFFMessageWithReply("Hello")
reply_msg = domain.send_immediate_msg_with_reply(msg)
