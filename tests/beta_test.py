from .conftest import SIGNALING_SERVER_PORT
import socket

def test_signaling_server(signaling_server):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        assert s.connect_ex(("127.0.0.1", SIGNALING_SERVER_PORT)) == 0