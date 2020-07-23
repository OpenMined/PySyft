from ...abstract.service import WorkerService
from ....message.search_message import SearchMessage, SearchMessageReply
from ..domain.domain import Domain as DomainWorker

class DomainSearchService(WorkerService):
    @staticmethod
    def process(
        worker: DomainWorker, msg: SearchMessage
    ) -> SearchMessageReply:
        # build a new route
        device, vm = None, None
        route = route(network = self.route.network, domain = self.route.domain,
            device = device, vm = vm)
        # tell them where to connect.
        route.configure_connection({})
        return SearchMessageReply(route=route, msg_id=self.id.value)

    @staticmethod
    def message_handler_types() -> List[type]:
        return [SearchMessage]
