import requests
import json
import syft as sy
from grid.websocket_client import WebsocketGridClient
from grid.utils import connect_all_nodes
import torch


SMPC_HOST_CHUNK = 4  # Minimum nodes required to host an encrypted model


class GridNetwork(object):
    """  The purpose of the Grid Network class is to control the entire communication flow by abstracting operational steps.

        Attributes:
            - gateway_url : network address to which you want to connect.
            - connected_grid_nodes : Grid nodes that are connected to the application.
    """

    def __init__(self, gateway_url):
        self.gateway_url = gateway_url

    def search(self, *query):
        """ Search a set of tags across the grid network.

            Arguments:
                query : A set of dataset tags.
            Returns:
                tensor_matrix : matrix of tensor pointers.
        """
        body = json.dumps({"query": list(query)})

        # Asks to grid gateway about dataset-tags
        response = requests.post(self.gateway_url + "/search", data=body)

        # List of nodes that contains the desired dataset
        match_nodes = json.loads(response.content)

        # Connect with grid nodes that contains the dataset and get their pointers
        tensor_set = []
        for node_id, node_url in match_nodes:
            worker = self.__connect_with_node(node_id, node_url)
            tensor_set.append(worker.search(query))
        return tensor_set

    def serve_encrypted_model(self, model):
        """ This method wiil choose some grid nodes at grid network to host an encrypted model.

            Args:
                model: Model to be hosted.
            Raise:
                RuntimeError : If grid network doesn't have enough workers to host an encrypted model.
        
        """
        # Model needs to be a plan
        if isinstance(model, sy.Plan):
            response = requests.get(self.gateway_url + "/choose-encrypted-model-host")
            hosts = json.loads(response.content)
            if (
                len(hosts) and len(hosts) % SMPC_HOST_CHUNK == 0
            ):  # Minimum workers chunk to share and host a model (3 to SMPC operations, 1 to host)
                for i in range(0, len(hosts), SMPC_HOST_CHUNK):
                    # Connect with SMPC Workers
                    smpc_end_interval = i + 2
                    smpc_workers_info = hosts[i:smpc_end_interval]
                    smpc_workers = []
                    for worker in smpc_workers_info:
                        smpc_workers.append(self.__connect_with_node(*worker))

                    # Connect with crypto provider
                    crypto_provider = self.__connect_with_node(
                        *hosts[smpc_end_interval]
                    )

                    # Connect with host worker
                    host = self.__connect_with_node(*hosts[smpc_end_interval + 1])

                    # Connect nodes to each other
                    model_nodes = smpc_workers + [crypto_provider, host]
                    connect_all_nodes(model_nodes)

                    # SMPC Share
                    model.fix_precision().share(
                        *smpc_workers, crypto_provider=crypto_provider
                    )
                    # Host model
                    p_model = model.send(host)

                    # Save model pointer
                    host.serve_model(p_model, model_id=model.id, mpc=True)

                    for node in model_nodes:
                        node.close()
                    smpc_initial_interval = i  # Initial index of next chunk
            # If host's length % SMPC_HOST_CHUNK != 0 or length == 0
            else:
                raise RuntimeError("Not enough workers to host an encrypted model!")
        # If model isn't a plan
        else:
            raise RuntimeError("Model needs to be a plan to be encrypted!")

    def serve_model(
        self,
        model,
        model_id,
        allow_remote_inference: bool = False,
        allow_download: bool = False,
    ):

        """ This method will choose one of grid nodes registered in the grid network to host a plain text model.
            Args:
                model: Model to be hosted.
                model_id: Model's ID.
                allow_remote_inference: Allow workers to run inference in this model.
                allow_download: Allow workers to copy the model and run it locally.
        """
        # Perform a request to choose model's host
        response = requests.get(self.gateway_url + "/choose-model-host")
        hosts = json.loads(response.content)

        for host_id, host_address in hosts:
            # Host model
            host_worker = self.__connect_with_node(host_id, host_address)
            host_worker.serve_model(
                model,
                model_id=model_id,
                allow_download=allow_download,
                allow_remote_inference=allow_remote_inference,
                mpc=False,  # Unencrypted model
            )
            host_worker.close()

    def run_encrypted_inference(self, model_id, data, copy=True):
        """ Search for an encrypted model and perform inference.
            
            Args:
                model_id: Model's ID.
                data: Dataset to be shared/inferred.
                copy: Boolean flag to perform encrypted inference without lose plan.
            Returns:
                Tensor: Inference's result.
        """
        # Search for an encrypted model
        body = json.dumps({"model_id": model_id})

        response = requests.post(
            self.gateway_url + "/search-encrypted-model", data=body
        )

        match_nodes = json.loads(response.content)

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

            crypto_node = self.__connect_with_node(
                crypto_provider_id, crypto_provider_url
            )

            # Share your dataset to same SMPC Workers
            shared_data = data.fix_precision().share(
                *workers, crypto_provider=crypto_node
            )

            # Perform Inference
            fetched_plan = sy.hook.local_worker.fetch_plan(
                model_id, host_node, copy=copy
            )

            return fetched_plan(shared_data).get().float_prec()
        else:
            raise RuntimeError("Model not found on Grid Network!")

    def run_remote_inference(self, model_id, data):
        """ This method will search for a specific model registered on grid network, if found,
            It will run inference.
            Args:
                model_id : Model's ID.
                dataset : Data used to run inference.
            Returns:
                Tensor : Inference's result.
        """
        worker = self.query_model(model_id)
        if worker:
            response = worker.run_remote_inference(model_id=model_id, data=data)
            worker.close()
            return torch.tensor(response)
        else:
            raise RuntimeError("Model not found on Grid Network!")

    def query_model(self, model_id):
        """ This method will search for a specific model registered on grid network, if found,
            It will return all grid nodes that contains the desired model.
            Args:
                model_id : Model's ID.
                data : Data used to run inference.
            Returns:
                workers : List of workers that contains the desired model.
        """
        # Search for a model
        body = json.dumps({"model_id": model_id})

        response = requests.post(self.gateway_url + "/search-model", data=body)

        match_nodes = json.loads(response.content)
        if len(match_nodes):
            node_id, node_url = match_nodes[0]  # Get the first node
            worker = self.__connect_with_node(node_id, node_url)
        else:
            worker = None
        return worker

    def __connect_with_node(self, node_id, node_url):
        if node_id not in sy.hook.local_worker._known_workers:
            worker = WebsocketGridClient(sy.hook, node_url, node_id)
        else:
            # There is already a connection to this node
            worker = sy.hook.local_worker._known_workers[node_id]
            worker.connect()
        return worker

    def disconnect_nodes(self):
        for node in sy.hook.local_worker._known_workers:
            if isinstance(
                sy.hook.local_worker._known_workers[node], WebsocketGridClient
            ):
                sy.hook.local_worker._known_workers[node].close()
