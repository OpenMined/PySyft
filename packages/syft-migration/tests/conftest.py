import pytest
from mocks import mock_registry

from syft_migration import MigrationRegistry, MigrationService


@pytest.fixture
def registry() -> MigrationRegistry:
    return mock_registry


@pytest.fixture
def service() -> MigrationService:
    return MigrationService(mock_registry)
