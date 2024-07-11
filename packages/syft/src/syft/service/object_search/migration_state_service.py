# stdlib

# relative
from syft.types.errors import SyftException
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
    stash: SyftMigrationStateStash

    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.stash = SyftMigrationStateStash(store=store)

    @service_method(path="migration", name="get_version")
    def get_version(
        self, context: AuthedServiceContext, canonical_name: str
    ) -> int:
        """Search for the metadata for an object."""

        migration_state = self.stash.get_by_name(
            canonical_name=canonical_name, credentials=context.credentials
        ).unwrap()

        if migration_state is None:
            raise SyftException(
                message=f"No migration state exists for canonical name: {canonical_name}"
            )

        return migration_state.current_version

    @service_method(path="migration", name="get_state")
    def get_state(
        self, context: AuthedServiceContext, canonical_name: str
    ) -> bool:
        return self.stash.get_by_name(
            canonical_name=canonical_name, credentials=context.credentials
        ).unwrap()

    @service_method(path="migration", name="register_migration_state")
    def register_migration_state(
        self,
        context: AuthedServiceContext,
        current_version: int,
        canonical_name: str,
    ) -> SyftObjectMigrationState:
        obj = SyftObjectMigrationState(
            current_version=current_version, canonical_name=canonical_name
        )
        return self.stash.set(migration_state=obj, credentials=context.credentials).unwrap()
