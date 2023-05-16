# third party
import pytest


@pytest.fixture
def numpy_syft_instance(guest_client):
    return guest_client.api.lib.numpy
