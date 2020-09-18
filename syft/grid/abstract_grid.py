import torch

from typing import Union
from typing import List
from typing import Any
from typing import Tuple
from typing import Dict

from abc import ABC, abstractmethod

from syft.grid.clients.data_centric_fl_client import DataCentricFLClient  # noqa: F401


class AbstractGrid(ABC):
    def __init__(self):
        self.SMPC_HOST_CHUNK = 4  # (1 host, 2 shares, 1 crypto_provider)

    @abstractmethod
    def search(self, *query: Union[str]) -> Dict[Any, Any]:
        raise NotImplementedError

    @abstractmethod
    def serve_model(
        self,
        model,
        id: str,
        mpc: bool = False,
        allow_remote_inference: bool = False,
        allow_download: bool = False,
        n_replica: int = 1,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def query_model_hosts(
        self, id: str, mpc: bool = False
    ) -> Union["DataCentricFLClient", Tuple["DataCentricFLClient"]]:
        raise NotImplementedError

    @abstractmethod
    def run_remote_inference(self, id: str, data: torch.Tensor, mpc: bool = False) -> torch.Tensor:
        raise NotImplementedError

    def _connect_all_nodes(self, nodes: Tuple[Any], node_type: Any) -> None:
        """Connect all nodes to each other.
        Args:
            nodes: A tuple of grid clients.
        """
        if self._check_node_type(nodes, node_type):  # Avoid connect local workers (Virtual Workers)
            for i in range(len(nodes)):
                for j in range(i):
                    node_i, node_j = nodes[i], nodes[j]
                    node_i.connect_nodes(node_j)
                    node_j.connect_nodes(node_i)

    def _check_node_type(self, grid_workers: List[Any], node_type: Any) -> bool:
        """Private method used to verify if workers used by grid network are exactly what we expect.

        Returns:
            result : Boolean result.
        """
        return all(isinstance(node, node_type) for node in grid_workers)
