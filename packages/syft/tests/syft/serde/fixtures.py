# third party
import pytest


@pytest.fixture
def numpy_syft_instance(guest_client):
    yield guest_client.api.lib.numpy
