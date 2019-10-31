from .models import GridNodes
from .models import db


def register_new_node(node_id, node_address):
    """ Register new grid node at grid network.
        Args:
            node_id (str) : Grid Node ID.
            address (str) : Grid Node Address.
    """
    registered = False
    if node_id not in connected_nodes().keys():
        new_node = GridNodes(id=node_id, address=node_address)
        db.session.add(new_node)
        db.session.commit()
        registered = True
    return registered


def delete_node(node_id, node_address):
    """ Delete a grid node at grid network.
        Args:
            node_id (str) : Grid Node ID.
            address (str) : Grid Node Address.
    """
    deleted = False
    if node_id in connected_nodes().keys():
        node_to_delete = GridNodes.query.filter_by(
            id=node_id, address=node_address
        ).first()
        db.session.delete(node_to_delete)
        db.session.commit()
        deleted = True

    return deleted


def connected_nodes():
    """ Retrieve all grid nodes connected on grid network.

        Returns:
            nodes_dict (Dictionary) : Dictionary{node_id:node_address} with connected grid nodes. 
    """
    nodes = GridNodes.query.all()
    nodes_dict = {}
    for node in nodes:
        nodes_dict[node.id] = node.address
    return nodes_dict
