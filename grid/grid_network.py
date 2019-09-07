import requests
import json
import syft as sy
from grid.websocket_client import WebsocketGridClient
import torch


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
            tensor_set.append(worker.search(*query))
        return tensor_set

    def serve_model(self, model, model_id):
        """ This method will choose one of grid nodes registered in the grid network to host a plain text model.
            Args:
                model : Model to be hosted.
                model_id : Model's ID.
        """
        # Perform a request to choose model's host
        response = requests.get(self.gateway_url + "/choose-model-host")
        hosts = json.loads(response.content)

        for host_id, host_address in hosts:
            # Host model
            host_worker = self.__connect_with_node(host_id, host_address)
            host_worker.serve_model(model, model_id=model_id)
            host_worker.disconnect()

    def run_inference(self, model_id, dataset):
        """ This method will search for a specific model registered on grid network, if found,
            It will run inference.
            Args:
                model_id : Model's ID.
                dataset : Data used to run inference.
            Returns:
                inference : result of data inference
        """
        worker = self.query_model(model_id)
        response = worker.run_inference(model_id=model_id, data=dataset)
        worker.disconnect()
        return torch.tensor(response["prediction"])

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
            worker.connect()
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
                sy.hook.local_worker._known_workers[node].disconnect()
