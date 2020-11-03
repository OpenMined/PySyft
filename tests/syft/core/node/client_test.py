# third party
import pytest

# syft absolute
import syft as sy


@pytest.mark.asyncio
async def test_client_from_metadata() -> None:

    domain = sy.Domain(name="duet")

    client_metadata = domain.get_metadata_for_client()

    spec_location, name, id = sy.DomainClient.deserialize_client_metadata_from_node(
        metadata=client_metadata.serialize()
    )

    assert domain.domain == spec_location
    assert name == "duet"
    assert id == domain.id
