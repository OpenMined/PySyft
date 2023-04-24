# third party
import pytest

# syft absolute
from syft.node.credentials import SyftSigningKey
from syft.node.credentials import SyftVerifyKey
from syft.service.action.action_graph import InMemoryGraphConfig


@pytest.fixture
def verify_key() -> SyftVerifyKey:
    signing_key = SyftSigningKey.generate()
    verify_key: SyftVerifyKey = signing_key.verify_key
    return verify_key


@pytest.fixture
def in_mem_graph_config() -> InMemoryGraphConfig:
    return InMemoryGraphConfig()
