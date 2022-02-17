# third party
import pytest

# syft absolute
import syft as sy
from syft import VirtualMachine
from syft import deserialize
from syft import serialize


@pytest.mark.asyncio
async def test_client_from_metadata(domain: sy.Domain) -> None:
    client_metadata = domain.get_metadata_for_client()

    spec_location, name, id = sy.DomainClient.deserialize_client_metadata_from_node(
        metadata=serialize(client_metadata)
    )

    assert domain.domain == spec_location
    assert name == "Alice"
    assert id == domain.id


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
