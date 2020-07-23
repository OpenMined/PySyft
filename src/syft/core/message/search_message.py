from .syft_message import SyftMessage, SyftMessageWithReply
from ..io.route import Route
from ..io.address import Address


class SearchMessage(SyftMessageWithReply):
    """
    when a network requests a route to a domain.
    this message is sent from the network.
    """
    def __init__(self, reply_to: Address, address: Address, msg_id: UID = None) -> None:
        super().__init__(reply_to=reply_to, address=address, msg_id=msg_id)

class SearchMessageReply(SyftMessage):
    """
    When a domain wants to inform a network of the route to a certain dataset
    this message is sent.
    """
    def __init__(self, address: Address, msg_id: UID = None,
        dataset: list = []) -> None:

        super().__init__(address=address, msg_id=msg_id)
        self.dataset = dataset
