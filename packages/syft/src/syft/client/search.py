# stdlib
from typing import List
from typing import Tuple
from typing import Union

# relative
from ..service.dataset.dataset import Dataset
from ..service.metadata.node_metadata import NodeMetadataJSON
from ..service.network.network_service import NodePeer
from ..types.uid import UID
from .client import SyftClient
from .registry import DomainRegistry


class SearchResults:
    def __init__(self, results: List[Tuple[SyftClient, List[Dataset]]]) -> None:
        self._dataset_client = {}
        self._datasets = []
        for pairs in results:
            client = pairs[0]
            datasets = list(pairs[1])
            for dataset in datasets:
                self._dataset_client[dataset.id] = client
                self._datasets.append(dataset)

    def __getitem__(self, key: Union[int, str, UID]) -> Dataset:
        if isinstance(key, int):
            return self._datasets[key]
        else:
            for dataset in self._datasets:
                if isinstance(key, UID):
                    if dataset.id == key:
                        return dataset
                elif isinstance(key, str):
                    if dataset.name == key:
                        return dataset
                    elif str(dataset.id) == key:
                        return dataset
        raise KeyError

    def __repr__(self) -> str:
        return str(self._datasets)

    def _repr_html_(self) -> str:
        return self._datasets._repr_html_()

    def client_for(self, key: Union[Dataset, int, str, UID]) -> SyftClient:
        if isinstance(key, Dataset):
            dataset = key
        else:
            dataset = self.__getitem__(key)
        return self._dataset_client[dataset.id]


class Search:
    def __init__(self, domains: DomainRegistry):
        self.domains = domains.online_domains

    @staticmethod
    def __search_one_node(
        peer_tuple: Tuple[NodePeer, NodeMetadataJSON], name: str
    ) -> List[Dataset]:
        try:
            peer, _ = peer_tuple
            client = peer.guest_client
            results = client.api.services.dataset.search(name=name)
            return (client, results)
        except:  # noqa
            return (None, [])

    def __search(self, name: str) -> List[Tuple[SyftClient, List[Dataset]]]:
        results = [
            self.__search_one_node(peer_tuple, name) for peer_tuple in self.domains
        ]

        # filter out SyftError
        filtered = ((client, result) for client, result in results if result)
        return filtered

    def search(self, name: str) -> SearchResults:
        return SearchResults(self.__search(name))
