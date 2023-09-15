# stdlib
from typing import Union

# relative
from ...serde.serializable import serializable
from ...store.document_store import DocumentStore
from ..context import AuthedServiceContext
from ..response import SyftError
from ..service import AbstractService
from ..service import service_method
from .object_migration_state import SyftMigrationStateStash
from .object_migration_state import SyftObjectMigrationState


@serializable()
class MigrateStateService(AbstractService):
    store: DocumentStore
    stash: SyftObjectMigrationState

    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.stash: SyftMigrationStateStash = SyftMigrationStateStash(store=store)

    @service_method(path="migration", name="get_version")
    def get_version(
        self, context: AuthedServiceContext, canonical_name: str
    ) -> Union[int, SyftError]:
        """Search for the metadata for an object."""

        result = self.stash.get_by_name(
            canonical_name=canonical_name, credentials=context.credentials
        )

        if result.is_err():
            return SyftError(message=f"{result.err()}")

        migration_state = result.ok()

        if migration_state is None:
            return SyftError(
                message=f"No migration state exists for canonical name: {canonical_name}"
            )

        return migration_state.current_version
