from . import base_worker
from .. import channels
from ..lib import strings

from ..services.passively_broadcast_membership import PassivelyBroadcastMembershipService
from ..services.listen_for_openmined_nodes import ListenForOpenMinedNodesService

class BaseClient(base_worker.GridWorker):

    def __init__(self,min_om_nodes=1,known_workers=list(),include_github_known_workers=True):
        super().__init__()
        self.progress = {}

        self.services = {}

        self.services['listen_for_openmined_nodes'] = ListenForOpenMinedNodesService(self,min_om_nodes,include_github_known_workers)
        # self.listen_for_openmined_nodes(min_om_nodes,include_github_known_workers)
    
    def listen(self):
        self.services['listen_for_openmined_nodes'].listen_for_openmined_nodes()