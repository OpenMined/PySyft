# syft relative
from ...core.node.network.client import NetworkClient
from ..connections.http_connection import HTTPConnection
from .signaling_client import SignalingClient


def register(
    url: str = "http://ec2-18-221-97-68.us-east-2.compute.amazonaws.com:5000",
) -> SignalingClient:
    client = SignalingClient(
        url=url, conn_type=HTTPConnection, client_type=NetworkClient
    )
    return client
