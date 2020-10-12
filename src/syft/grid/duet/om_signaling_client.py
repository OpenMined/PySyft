# syft relative
from ...core.node.network.client import NetworkClient
from ..connections.http_connection import HTTPConnection
from .signaling_client import SignalingClient

WebRTC_HOST = (
    "http://ec2-18-191-23-46.us-east-2.compute.amazonaws.com:5000"  # noqa: F811
)


def register(
    url: str = WebRTC_HOST,
) -> SignalingClient:
    client = SignalingClient(
        url=url, conn_type=HTTPConnection, client_type=NetworkClient
    )
    return client
