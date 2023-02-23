# stdlib
from typing import List
from typing import Tuple

# relative
from .core.node.common.client import Client
from .core.node.new.dataset import Dataset
from .registry import NetworkRegistry


class SearchResults:
    def __init__(self, results: List[Tuple[Client, List[Dataset]]]):
        self._results = results
        self._clients = (client for client, _ in results)
        self._datasets = sum((datasets for _, datasets in results), start=[])

    def __getitem__(self, idx: int) -> Dataset:
        return self._datasets[idx]

    def __repr__(self) -> str:
        return str(self._results)


class Search:
    def __init__(self, gateways: NetworkRegistry):
        self.gateways = gateways

    @staticmethod
    def __search_one_node(client: Client, name: str) -> List[Dataset]:
        try:
            return client.api.services.dataset.search(name=name)
        except:  # noqa
            return []

    def __search(self, name: str) -> List[Tuple[Client, List[Dataset]]]:
        results = (
            (client, self.__search_one_node(client, name)) for client in self.gateways
        )

        # filter out SyftError
        filtered = ((client, result) for client, result in results if result)

        return list(filtered)

    def search(self, name: str) -> SearchResults:
        return SearchResults(self.__search(name))
