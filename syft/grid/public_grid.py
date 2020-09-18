import torch
import requests
import json

from typing import Union
from typing import List
from typing import Any
from typing import Tuple
from typing import Dict

# Syft imports
from syft.grid.abstract_grid import AbstractGrid
from syft.grid.clients.data_centric_fl_client import DataCentricFLClient
from syft.execution.plan import Plan
from syft.codes import GATEWAY_ENDPOINTS


class PublicGridNetwork(AbstractGrid):
    def __init__(self, hook, gateway_url: str):
        super().__init__()
        self.hook = hook
        self.gateway_url = gateway_url

    def search(self, *query: Union[str]) -> Dict[Any, Any]:
        """Search a set of tags across the grid network.

        Args:
            query : A set of dataset tags.
        Returns:
            tensor_results : matrix of tensor pointers.
        """
        # Ask gateway about desired tags
        body = json.dumps({"query": list(query)})
        match_nodes = self._ask_gateway(requests.post, GATEWAY_ENDPOINTS.SEARCH_TAGS, body)

        # Connect with grid nodes that contains the dataset and get their pointers
        tensor_results = {}
        for node_id, node_url in match_nodes:
            worker = self.__connect_with_node(node_id, node_url)
            tensor_results[node_id] = worker.search(query)
        return tensor_results

    def serve_model(
        self,
        model,
        id: Union[str, int],
        mpc: bool = False,
        allow_remote_inference: bool = False,
        allow_download: bool = False,
        n_replica: int = 1,
    ) -> None:
        """Choose n (number of replicas defined at gateway) grid nodes registered
        in the grid network to host a model.

        Args:
            model : Model to be hosted.
            id : Model's ID.
            mpc : Boolean flag to serve plan models in an encrypted/unencrypted format.
            allow_remote_inference : Allow workers to run inference in this model.
            allow_download : Allow workers to copy the model and run it locally.
        """
        if not mpc:
            self._serve_unencrypted_model(model, id, allow_remote_inference, allow_download)
        else:
            self._serve_encrypted_model(model)

    def query_model_hosts(
        self, id: str, mpc: bool = False
    ) -> Union["DataCentricFLClient", Tuple["DataCentricFLClient"]]:
        """This method will search for a specific model registered on grid network, if found,
        It will return all grid nodes that contains the desired model.

        Args:
            id : Model's ID.
            mpc : Boolean flag to search plan models in an encrypted/unencrypted format.
        Returns:
            workers : Worker / list of workers that contains the desired model.
        Raises:
            RuntimeError : If grid network doesn't have enough workers to host
            an encrypted model, or if model isn't a plan.
        """
        if not mpc:
            return self._query_unencrypted_models(id)
        else:
            return self._query_encrypted_models(id)

    def run_remote_inference(self, id: str, data: torch.Tensor, mpc: bool = False) -> torch.Tensor:
        """This method will search for a specific model registered on the grid network, if found,
        It will run inference.

        Args:
            id : Model's ID.
            dataset : Data used to run inference.
            mpc : Boolean flag to run encrypted/unencrypted inferences.
        Returns:
            Tensor : Inference's result.
        Raises:
            RuntimeError: If model id not registered on the grid network.
        """
        if not mpc:
            return self._run_unencrypted_inference(id, data)
        else:
            return self._run_encrypted_inference(id, data)

    def _serve_unencrypted_model(
        self, model, id, allow_remote_inference: bool, allow_download: bool
    ) -> None:
        """This method will choose one of grid nodes registered in the grid network
        to host a plain text model.

        Args:
            model: Model to be hosted.
            id: Model's ID.
            allow_remote_inference: Allow workers to run inference in this model.
            allow_download: Allow workers to copy the model and run it locally.
        """
        hosts = self._ask_gateway(requests.get, GATEWAY_ENDPOINTS.SELECT_MODEL_HOST)

        for host_id, host_address in hosts:
            # Host model
            host_worker = self.__connect_with_node(host_id, host_address)
            host_worker.serve_model(
                model,
                model_id=id,
                allow_download=allow_download,
                allow_remote_inference=allow_remote_inference,
                mpc=False,
            )
        host_worker.close()

    def _serve_encrypted_model(self, model) -> None:
        """This method wiil choose some grid nodes at grid network to host an encrypted model.

        Args:
            model: Model to be hosted.
        Raises:
            RuntimeError : If grid network doesn't have enough workers to host
            an encrypted model, or if model isn't a plan.
        """
        # Model needs to be a plan
        if isinstance(model, Plan):
            hosts = self._ask_gateway(requests.get, GATEWAY_ENDPOINTS.SELECT_ENCRYPTED_MODEL_HOSTS)
            if (
                len(hosts) and len(hosts) % self.SMPC_HOST_CHUNK == 0
            ):  # Minimum workers chunk to share and host a model (3 to SMPC operations, 1 to host)
                for i in range(0, len(hosts), self.SMPC_HOST_CHUNK):

                    # Connect with SMPC Workers
                    smpc_end_interval = i + 2
                    smpc_workers_info = hosts[i:smpc_end_interval]
                    smpc_workers = []
                    for worker in smpc_workers_info:
                        smpc_workers.append(self.__connect_with_node(*worker))

                    # Connect with crypto provider
                    crypto_provider = self.__connect_with_node(*hosts[smpc_end_interval])

                    # Connect with host worker
                    host = self.__connect_with_node(*hosts[smpc_end_interval + 1])

                    # Connect nodes to each other
                    model_nodes = smpc_workers + [crypto_provider, host]
                    self._connect_all_nodes(model_nodes, DataCentricFLClient)

                    # SMPC Share
                    model.fix_precision().share(*smpc_workers, crypto_provider=crypto_provider)

                    # Host model
                    p_model = model.send(host)

                    # Save model pointer
                    host.serve_model(p_model, model_id=model.id, mpc=True)

                    for node in model_nodes:
                        node.close()
            # If host's length % SMPC_HOST_CHUNK != 0 or length == 0
            else:
                raise RuntimeError("Not enough workers to host an encrypted model!")
        # If model isn't a plan
        else:
            raise RuntimeError("Model needs to be a plan to be encrypted!")

    def _query_unencrypted_models(self, id) -> "DataCentricFLClient":
        """Search for a specific model registered on grid network, if found,
        It will return the first node that contains the desired model.

        Args:
            id : Model's ID.
        Returns:
            worker : worker that contains the desired model.
        """
        # Search for a model
        body = json.dumps({"model_id": id})
        match_nodes = self._ask_gateway(requests.post, GATEWAY_ENDPOINTS.SEARCH_MODEL, body)

        for node_id, node_url in match_nodes:
            # Return the first node that stores the desired model
            return self.__connect_with_node(node_id, node_url)

    def _query_encrypted_models(self, id) -> List["DataCentricFLClient"]:
        """Search for a specific encrypted model registered on grid network, if found,
        It will return the first node that hosts the desired model and mpc shares.

        Args:
            id : Model's ID.
        Returns:
            workers : List of workers that contains the desired mpc model.
        """
        # Search for an encrypted model
        body = json.dumps({"model_id": id})
        match_nodes = self._ask_gateway(
            requests.post, GATEWAY_ENDPOINTS.SEARCH_ENCRYPTED_MODEL, body
        )

        if len(match_nodes):
            # Host of encrypted plan
            node_id = list(match_nodes.keys())[0]  # Get the first one
            node_address = match_nodes[node_id]["address"]

            # Workers with SMPC parameters tensors
            worker_infos = match_nodes[node_id]["nodes"]["workers"]
            crypto_provider = match_nodes[node_id]["nodes"]["crypto_provider"]

            # Connect with host node
            host_node = self.__connect_with_node(node_id, node_address)

            # Connect with SMPC Workers
            workers = []
            for worker_id, worker_address in worker_infos:
                workers.append(self.__connect_with_node(worker_id, worker_address))

            # Connect with SMPC crypto provider
            crypto_provider_id = crypto_provider[0]
            crypto_provider_url = crypto_provider[1]

            crypto_node = self.__connect_with_node(crypto_provider_id, crypto_provider_url)

            # Connect nodes
            nodes = workers + [host_node, crypto_node]
            self._connect_all_nodes(tuple(nodes), DataCentricFLClient)

            return (host_node, workers, crypto_node)
        else:
            raise RuntimeError("Model not found on Grid Network!")

    def _run_unencrypted_inference(self, id, data) -> torch.Tensor:
        """Search for an unencrypted model and perform data inference.

        Args:
            id: Model's ID.
            data: Dataset to be inferred.
        Returns:
            Tensor: Inference's result.
        Raises:
            RuntimeError: If model if not found.
        """
        worker = self.query_model_hosts(id)
        if worker:
            response = worker.run_remote_inference(model_id=id, data=data)
            worker.close()
            return torch.tensor(response)
        else:
            raise RuntimeError("Model not found on Grid Network!")

    def _run_encrypted_inference(self, id, data, copy=True):
        """Search for an encrypted model and perform inference.

        Args:
            model_id: Model's ID.
            data: Dataset to be shared/inferred.
            copy: Boolean flag to perform encrypted inference without lose plan.
        Returns:
            Tensor: Inference's result.
        Raises:
            RuntimeError: If model id not found.
        """
        # Get model's host / mpc shares
        host_node, smpc_workers, crypto_provider = self._query_encrypted_models(id)

        # Share your dataset to same SMPC Workers
        shared_data = data.fix_precision().share(*smpc_workers, crypto_provider=crypto_provider)

        # Perform Inference
        fetched_plan = self.hook.local_worker.fetch_plan(id, host_node, copy=copy)
        return fetched_plan(shared_data).get().float_prec()

    def __connect_with_node(self, node_id, node_url):
        if node_id not in self.hook.local_worker._known_workers:
            worker = DataCentricFLClient(self.hook, node_url)
        else:
            # There is already a connection to this node
            worker = self.hook.local_worker._known_workers[node_id]
            worker.connect()
        return worker

    def _ask_gateway(self, request_method, endpoint: str, body: Dict = {}):
        response = request_method(self.gateway_url + endpoint, data=body)
        return json.loads(response.content)
