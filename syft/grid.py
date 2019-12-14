import random
import torch
from collections import Counter
from typing import Any
from typing import Tuple
from typing import Counter as CounterType
from typing import Dict
from typing import Union

# Syft imports
from syft.workers.node_client import NodeClient
from syft.messaging.plan.plan import Plan
from syft.frameworks.torch.tensors.interpreters.additive_shared import AdditiveSharingTensor


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
        id: str,
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

    def run_remote_inference(self, id: str, data: torch.Tensor):
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

    def query_model(self, id: str):
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

    def host_encrypted_model(self, model, id: str, n_shares: int = 3):
        """ This method wiil choose some grid nodes at grid network to host an encrypted model.

            Args:
                model: Model to be hosted.
                id: Model's id.
                n_shares: number of workers used by MPC protocol.
            Raise:
                RuntimeError : If grid network doesn't have enough workers
                to host an encrypted model or if model is not a plan.
        """
        # Verify if this network have enough workers.
        if n_shares > len(self.workers):
            raise RuntimeError("Not enough nodes!")
        elif n_shares < 3:
            raise RuntimeError("Not enough shares to perform MPC operations!")
        else:
            # Select N workers in your set of workers.
            nodes = random.sample(self.workers, n_shares)

        # Model needs to be a plan
        if isinstance(model, Plan):
            host = nodes[0]  # Host
            mpc_nodes = nodes[:-1]  # Shares
            crypto_provider = nodes[-1]  # Crypto Provider

            # SMPC Share
            model.fix_precision().share(*mpc_nodes, crypto_provider=crypto_provider)

            # Host model
            model.send(host)

        # If model isn't a plan
        else:
            raise RuntimeError("Model needs to be a plan to be encrypted!")

    def query_encrypted_model(self, id: str):
        """ Search for an encrypted model and return its mpc nodes.

            Args:
                id: Model's ID.
            Returns:
                Tuple : Tuple structure containing Host, MPC Nodes and crypto provider.
            Raises:
                RuntimeError: If model id not found.
        """
        # Search for Pointer Plan
        model = list(filter(lambda x: x._objects.get(id), self.workers))

        # If it's registered on grid nodes.
        if len(model):
            host = model[0].owner  # Get the host of the first model found.
            mpc_nodes = set()
            crypto_provider = None

            # Check every state used by this plan
            for state_id in model.state.state_ids:
                obj = host._objects.get(state_id)

                # Decrease in Tensor Hierarchy.
                # (we want be a AdditiveSharingTensor to recover workers/crypto_provider addresses)
                while not isinstance(obj, AdditiveSharingTensor):
                    obj = obj.child

                # Get a list of mpc nodes.
                nodes = map(lambda x: (x, host._known_workers.get(x)), obj.child.keys(),)

                mpc_nodes.update(set(nodes))

                if obj.crypto_provider:
                    crypto_provider = obj.crypto_provider

                return (host, mpc_nodes, crypto_provider)
        else:
            raise RuntimeError("Model ID not found!")

    def run_encrypted_inference(self, id: str, data, copy=True):
        """ Search for an encrypted model and perform inference.
            
            Args:
                model_id: Model's ID.
                data: Dataset to be shared/inferred.
                copy: Boolean flag to perform encrypted inference without lose plan.
            Returns:
                Tensor: Inference's result.
            Raises:
                RuntimeError: If model id not found.
        """
        host, mpc_nodes, crypto_provider = self.query_encrypted_model(id)

        # Share your dataset to same SMPC Workers
        shared_data = data.fix_precision().share(*mpc_nodes, crypto_provider=crypto_provider)

        # Perform Inference
        fetched_plan = self.hook.local_worker.fetch_plan(id, host, copy=copy)

        return fetched_plan(shared_data).get().float_prec()
