import random
import torch
from collections import Counter
from typing import Any
from typing import Tuple
from typing import Counter as CounterType
from typing import Dict
from typing import Union
from syft.workers.node_client import NodeClient


class GridNetwork:
    def __init__(self, *workers):
        self.workers = list(workers)
        self._connect_all_nodes(self.workers)

    def _connect_all_nodes(self, nodes: list):
        """Connect all nodes to each other.

        Args:
            nodes: A tuple of grid clients.
        """
        if all(isinstance(NodeClient, type(node)) for node in nodes):
            for i in range(len(nodes)):
                for j in range(i):
                    node_i, node_j = nodes[i], nodes[j]
                    node_i.connect_nodes(node_j)
                    node_j.connect_nodes(node_i)

    def search(
        self, *query, verbose: bool = True, return_counter: bool = True
    ) -> Union[Tuple[Dict[Any, Any], CounterType], Dict[Any, Any]]:
        """Searches over a collection of workers, returning pointers to the results
        grouped by worker.
        """

        tag_counter: CounterType[int] = Counter()
        result_counter = 0

        results = {}
        for worker in self.workers:

            worker_tag_ctr: CounterType[int] = Counter()

            worker_results = worker.search(query)

            if len(worker_results) > 0:
                results[worker.id] = worker_results

                for result in worker_results:
                    for tag in result.tags:
                        tag_counter[tag] += 1
                        worker_tag_ctr[tag] += 1

                if verbose:
                    tags = str(worker_tag_ctr.most_common(3))
                    print(f"Found {str(len(worker_results))} results on {str(worker)} - {tags}")

                result_counter += len(worker_results)

        if verbose:
            print("\nFound " + str(result_counter) + " results in total.")
            print("\nTag Profile:")
            for tag, count in tag_counter.most_common():
                print("\t" + tag + " found " + str(count))

        if return_counter:
            return results, tag_counter
        else:
            return results

    def serve_model(
        self,
        model,
        id,
        allow_remote_inference: bool = False,
        allow_download: bool = False,
        n_replica: int = 1,
    ):

        """ Choose some node(s) on grid network to host a unencrypted model.
            Args:
                model: Model to be hosted.
                id: Model's ID.
                allow_remote_inference: Allow to run inference remotely.
                allow_download: Allow to copy the model and run it locally.
                n_replica: Number of copies distributed through grid network.
            Raises:
                RuntimeError: If grid network doesn't have enough nodes to replicate the model.
        """
        if n_replica > len(self.workers):
            raise RuntimeError("Not enough nodes!")
        else:
            nodes = random.sample(self.workers, n_replica)

        for node in nodes:
            # Host model
            node.serve_model(
                model,
                model_id=id,
                allow_download=allow_download,
                allow_remote_inference=allow_remote_inference,
            )

    def run_remote_inference(self, id, data):
        """ Search for a specific model registered on grid network, if found,
            It will run inference.
            Args:
                id : Model's ID.
                dataset : Data used to run inference.
            Returns:
                Tensor : Inference's result.
        """
        node = self.query_model(id)
        if node:
            response = node.run_remote_inference(model_id=id, data=data)
            return torch.tensor(response)
        else:
            raise RuntimeError("Model not found on Grid Network!")

    def query_model(self, id):
        """ Search for a specific model registered on grid network, if found,
            It will return all grid nodes that contains the desired model.
            Args:
                id : Model's ID.
                data : Data used to run inference.
            Returns:
                workers : List of workers that contains the desired model.
        """
        for node in self.workers:
            if id in node.models:
                return node
