from .nodes import GridNodes, db


class NetworkManager:
    """NetworkManager registers, deletes, and retrives grid nodes in the PyGrid
    network."""

    def __init__(self):
        pass

    def register_new_node(self, node_id, node_address):
        """Register a new grid node in the grid network.

        Args:
            node_id (str): Grid Node ID.
            node_address (str): Grid Node Address.
        """
        registered = False
        if node_id not in self.connected_nodes().keys():
            new_node = GridNodes(id=node_id, address=node_address)
            db.session.add(new_node)
            db.session.commit()
            registered = True
        return registered

    def delete_node(self, node_id, node_address):
        """Delete a grid node in the grid network.

        Args:
            node_id (str): Grid Node ID.
            node_address (str): Grid Node Address.
        """
        deleted = False
        if node_id in self.connected_nodes().keys():
            node_to_delete = GridNodes.query.filter_by(
                id=node_id, address=node_address
            ).first()
            db.session.delete(node_to_delete)
            db.session.commit()
            deleted = True

        return deleted

    def connected_nodes(self):
        """Retrieve all grid nodes connected in the grid network.

        Returns:
            dict: Connected grid nodes in the key-value pair of node_id: node_address.
        """
        nodes = GridNodes.query.all()
        nodes_dict = {}
        for node in nodes:
            nodes_dict[node.id] = node.address
        return nodes_dict
