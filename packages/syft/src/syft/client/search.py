# stdlib
from concurrent.futures import ThreadPoolExecutor

# third party
from IPython.display import display

# relative
from ..service.dataset.dataset import Dataset
from ..service.metadata.node_metadata import NodeMetadataJSON
from ..service.network.network_service import NodePeer
from ..service.response import SyftWarning
from ..types.uid import UID
from .client import SyftClient
from .registry import DomainRegistry


class SearchResults:
    def __init__(self, results: list[tuple[SyftClient, list[Dataset]]]) -> None:
        self._dataset_client = {}
        self._datasets = []
        for pairs in results:
            client = pairs[0]
            datasets = list(pairs[1])
            for dataset in datasets:
                self._dataset_client[dataset.id] = client
                self._datasets.append(dataset)

    def __getitem__(self, key: int | str | UID) -> Dataset:
        if isinstance(key, int):
            return self._datasets[key]
        else:
            for dataset in self._datasets:
                if isinstance(key, UID):
                    if dataset.id == key:
                        return dataset
                elif isinstance(key, str):
                    if dataset.name == key or str(dataset.id) == key:
                        return dataset
        raise KeyError

    def __repr__(self) -> str:
        return str(self._datasets)

    def _repr_html_(self) -> str:
        return self._datasets._repr_html_()

    def client_for(self, key: Dataset | int | str | UID) -> SyftClient:
        if isinstance(key, Dataset):
            dataset = key
        else:
            dataset = self.__getitem__(key)
        return self._dataset_client[dataset.id]

    def __len__(self) -> int:
        return len(self._datasets)


class Search:
    def __init__(self, domains: DomainRegistry) -> None:
        self.domains: list[tuple[NodePeer, NodeMetadataJSON | None]] = (
            domains.online_domains
        )

    @staticmethod
    def __search_one_node(
        peer_tuple: tuple[NodePeer, NodeMetadataJSON], name: str
    ) -> tuple[SyftClient | None, list[Dataset]]:
        try:
            peer, node_metadata = peer_tuple
            client = peer.guest_client
            results = client.api.services.dataset.search(name=name)
            return (client, results)
        except Exception as e:  # noqa
            warning = SyftWarning(
                message=f"Got exception {e} at node {node_metadata.name}"
            )
            display(warning)
            return (None, [])

    def __search(self, name: str) -> list[tuple[SyftClient, list[Dataset]]]:
        with ThreadPoolExecutor(max_workers=20) as executor:
            # results: list[tuple[SyftClient | None, list[Dataset]]] = [
            #     self.__search_one_node(peer_tuple, name) for peer_tuple in self.domains
            # ]
            results: list[tuple[SyftClient | None, list[Dataset]]] = list(
                executor.map(
                    lambda peer_tuple: self.__search_one_node(peer_tuple, name),
                    self.domains,
                )
            )
        # filter out SyftError
        filtered = [(client, result) for client, result in results if client and result]

        return filtered

    def search(self, name: str) -> SearchResults:
        """
        Searches for a specific dataset by name.

        Args:
            name (str): The name of the dataset to search for.

        Returns:
            SearchResults: An object containing the search results.
        """
        return SearchResults(self.__search(name))
