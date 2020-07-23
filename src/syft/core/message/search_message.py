from .syft_message import SyftMessage, SyftMessageWithReply
from ..io.route import route, Route

class SearchMessage(SyftMessageWithReply):
    """
    when a network requests a route to a domain.
    this message is sent from the network.
    """
    def __init__(self, route: Route, msg_id: UID = None) -> None:
        super().__init__(route=route, msg_id=msg_id)

class SearchMessageReply(SyftMessage):
    """
    When a domain wants to inform a network of the route to a certain dataset
    this message is sent.
    """
    def __init__(self, route: Route, reply_to: Route, msg_id: UID = None,
        dataset: list = []) -> None:

        super().__init__(route=route, msg_id=msg_id, reply_to=reply_to)
        self.dataset = dataset
