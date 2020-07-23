from .syft_message import SyftMessage, SyftMessageWithReply
from ..io.route import route, Route
from ...common.token import Token


class SearchMessage(SyftMessageWithReply):
    """
    when a network requests a route to a domain.
    this message is sent from the network.
    """
    def __init__(self, reply_to: Route, token: Token = None, msg_id: UID = None) -> None:
        super().__init__(reply_to=reply_to, token=token, msg_id=msg_id)

class SearchMessageReply(SyftMessage):
    """
    When a domain wants to inform a network of the route to a certain dataset
    this message is sent.
    """
    def __init__(self, token: Token = None, msg_id: UID = None,
        dataset: list = []) -> None:

        super().__init__(token=token, msg_id=msg_id)
        self.dataset = dataset
