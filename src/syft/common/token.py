import jwt
from object import ObjectWithId

class Token(ObjectWithId):
    """
    A token encompases information that messages across different
    nodes can exchange, to maintain secure information.
    """
    def encode(self, secret, data_tags, private_routes, authorized_node):
        payload = {
            'data_tags': data_tags,
            'private_routes': private_routes,
            'authorized_node_id': authorized_node
        }
        return jwt.encode(payload, secret, algorithm='HS256')

    def decode(self, token, secret):
        return jwt.decode(token, secret, algorithms=['HS256'])
