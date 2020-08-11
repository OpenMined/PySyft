import syft as sy


def test_client_from_metadata():

    domain = sy.Domain(name="duet")

    client_metadata = domain.get_metadata_for_client()

    target_id, name, id = sy.DomainClient.deserialize_client_metadata_from_node(
        metadata=client_metadata
    )

    assert domain.domain == target_id
    assert name == "duet"
    assert id == domain.id
