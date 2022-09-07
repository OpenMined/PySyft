# syft absolute
import syft as sy
from syft import VirtualMachine
from syft import deserialize
from syft import serialize


def test_client_from_metadata(domain: sy.Domain) -> None:
    client_metadata = domain.get_metadata_for_client()

    assert domain.domain == client_metadata.node
    assert "Alice" == client_metadata.name
    assert domain.id == client_metadata.id


def test_client_serde() -> None:
    client = VirtualMachine(name="Bob").get_client()

    blob = serialize(client, to_bytes=True)
    recreated_client = deserialize(blob, from_bytes=True)
    assert client.name == recreated_client.name
    assert client.routes == recreated_client.routes
    assert client.network == recreated_client.network
    assert client.domain == recreated_client.domain
    assert client.device == recreated_client.device
    assert client.vm == recreated_client.vm
