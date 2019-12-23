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
            print(f"\nFound {str(result_counter)} results in total.")
            print("\nTag Profile:")
            for tag, count in tag_counter.most_common():
                print(f"\t {tag} found {str(count)}")

        if return_counter:
            return results, tag_counter
        else:
            return results

    def serve_model(
        self,
        model,
        id: str,
        mpc: bool = False,
        allow_remote_inference: bool = False,
        allow_download: bool = False,
        n_replica: int = 1,
    ):
        """ Choose some node(s) on grid network to host a unencrypted / encrypted model.
            Args:
                model: Model to be hosted.
                id: Model's ID.
                mpc: Boolean flag to host a plain text / encrypted model
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

        for i in range(len(nodes)):
            if not mpc:
                # Host model
                nodes[i].serve_model(
                    model,
                    model_id=id,
                    allow_download=allow_download,
                    allow_remote_inference=allow_remote_inference,
                )
            else:
                self._host_encrypted_model(model)

    def run_remote_inference(self, id: str, data: torch.Tensor, mpc: bool = False):
        """ Search for a specific model registered on grid network, if found,
            It will run inference.
            Args:
                id : Model's ID.
                dataset : Data used to run inference.
                mpc: Boolean flag to run a plain text / encrypted model
            Returns:
                Tensor : Inference's result.
        """
        if not mpc:
            node = self.query_model_host(id)
            if node:
                response = node.run_remote_inference(model_id=id, data=data)
                return torch.tensor(response)
            else:
                raise RuntimeError("Model not found on Grid Network!")
        else:
            return self._run_encrypted_inference(id, data)

    def query_model_host(self, id: str, mpc: bool = False):
        """ Search for node host from a specific model registered on grid network, if found,
            It will return the frist host/ set of hosts that contains the desired model.
            Args:
                id : Model's ID.
                data : Data used to run inference.
                mpc : Boolean flag to search for a plain text / encrypted model
            Returns:
                workers : First worker that contains the desired model.
        """
        # If it isn't a mpc model
        if not mpc:
            for node in self.workers:
                if id in node.models:
                    return node
        else:
            # MPC model
            return self._query_encrypted_model_hosts(id)

    def _host_encrypted_model(self, model, n_shares: int = 4):
        """ This method wiil choose some grid nodes at grid network to host an encrypted model.

            Args:
                model: Model to be hosted.
                n_shares: number of workers used by MPC protocol.
            Raise:
                RuntimeError : If grid network doesn't have enough workers
                to host an encrypted model or if model is not a plan.
        """
        # Verify if this network have enough workers.
        if n_shares > len(self.workers):
            raise RuntimeError("Not enough nodes!")
        elif n_shares < 4:
            raise RuntimeError("Not enough shares to perform MPC operations!")
        else:
            # Select N workers in your set of workers.
            nodes = random.sample(self.workers, n_shares)

        # Model needs to be a plan
        if isinstance(model, Plan):
            host = nodes[0]  # Host
            mpc_nodes = nodes[1:-1]  # Shares
            crypto_provider = nodes[-1]  # Crypto Provider

            # SMPC Share
            model.fix_precision().share(*mpc_nodes, crypto_provider=crypto_provider)

            # Host model
            p_model = model.send(host)

            # Save a pointer reference to this model in database.
            host.serve_model(
                p_model,
                model_id=model.id,
                allow_download=False,
                allow_remote_inference=False,
                mpc=True,
            )
        # If model isn't a plan
        else:
            raise RuntimeError("Model needs to be a plan to be encrypted!")

    def _query_encrypted_model_hosts(self, id: str):
        """ Search for an encrypted model and return its mpc nodes.

            Args:
                id: Model's ID.
            Returns:
                Tuple : Tuple structure containing Host, MPC Nodes and crypto provider.
            Raises:
                RuntimeError: If model id not found.
        """
        host = self.query_model_host(id)

        # If it's registered on grid nodes.
        if host:
            model = host.search(id)[0].get(deregister_ptr=False)
            mpc_nodes = set()
            crypto_provider = None

            # Check every state used by this plan
            for state_id in model.state.state_ids:
                hook = host.hook
                obj = hook.local_worker._objects.get(state_id)

                # Decrease in Tensor Hierarchy.
                # (we want be a AdditiveSharingTensor to recover workers/crypto_provider addresses)
                while not isinstance(obj, AdditiveSharingTensor):
                    obj = obj.child

                # Get a list of mpc nodes.
                nodes = map(lambda x: hook.local_worker._known_workers.get(x), obj.child.keys(),)

                mpc_nodes.update(set(nodes))

                if obj.crypto_provider:
                    crypto_provider = obj.crypto_provider

                return (host, mpc_nodes, crypto_provider)
        else:
            raise RuntimeError("Model ID not found!")

    def _run_encrypted_inference(self, id: str, data):
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
        host, mpc_nodes, crypto_provider = self._query_encrypted_model_hosts(id)

        # Share your dataset to same SMPC Workers
        shared_data = data.fix_precision().share(*mpc_nodes, crypto_provider=crypto_provider)

        # Perform Inference
        fetched_plan = host.hook.local_worker.fetch_plan(id, host, copy=True)

        return fetched_plan(shared_data).get().float_prec()

    def _connect_all_nodes(self, nodes: list):
        """Connect all nodes to each other.

        Args:
            nodes: A tuple of grid clients.
        """
        if all(isinstance(node, NodeClient) for node in nodes):
            for i in range(len(nodes)):
                for j in range(i):
                    node_i, node_j = nodes[i], nodes[j]
                    node_i.connect_nodes(node_j)
                    node_j.connect_nodes(node_i)
