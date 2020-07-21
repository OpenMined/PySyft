from .syft_message import SyftMessageWithReply

class SearchResponseMessage(SyftMessage):
    """
    when a network requests a route to a domain.
    this message is sent from the network.
    """
    def __init__(self, route: Route, msg_id: UID = None) -> None:
        super().__init__(route=route, msg_id=msg_id)

class SearchRequestMessage(SyftMessageWithReply):
    """
    When a domain wants to inform a network of the route to a certain dataset
    this message is sent.
    """
    def __init__(self, route: Route, reply_to: Route, msg_id: UID = None,
        dataset: list = []) -> None:

        super().__init__(route=route, msg_id=msg_id, reply_to=reply_to)
        self.dataset = dataset

    def process(self):
        # build a new route
        device, vm = None, None
        route = Route(network = self.route.network, domain = self.route.domain,
            device = device, vm = vm)
        # tell them where to connect.
        route.configure_connection({})
        return SearchResponseMessage(route=route, msg_id=self.id.value) 
