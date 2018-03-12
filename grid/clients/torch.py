from . import base
from ..services.torch.torch_service import TorchService


class TorchClient(base.BaseClient):
    def __init__(self,
                 min_om_nodes=1,
                 known_workers=list(),
                 include_github_known_workers=True,
                 verbose=True):
        super().__init__(
            min_om_nodes=min_om_nodes,
            known_workers=known_workers,
            include_github_known_workers=include_github_known_workers,
            verbose=verbose)

        self.services['torch_service'] = TorchService(self)
