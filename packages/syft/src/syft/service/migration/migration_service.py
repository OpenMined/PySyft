# stdlib

# stdlib

# stdlib

# third party
from result import Err
from result import Ok
from result import Result

# relative
from ...serde.serializable import serializable
from ...store.document_store import DocumentStore
from ...types.syft_object import SyftObject
from ..action.action_object import Action
from ..action.action_object import ActionObject
from ..context import AuthedServiceContext
from ..response import SyftError
from ..response import SyftSuccess
from ..service import AbstractService
from ..service import service_method
from ..user.user_roles import ADMIN_ROLE_LEVEL
from .object_migration_state import SyftMigrationStateStash
from .object_migration_state import SyftObjectMigrationState


@serializable()
class MigrationService(AbstractService):
    store: DocumentStore
    stash: SyftMigrationStateStash

    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.stash = SyftMigrationStateStash(store=store)

    @service_method(path="migration", name="get_version")
    def get_version(
        self, context: AuthedServiceContext, canonical_name: str
    ) -> int | SyftError:
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

    @service_method(path="migration", name="get_state")
    def get_state(
        self, context: AuthedServiceContext, canonical_name: str
    ) -> bool | SyftError:
        result = self.stash.get_by_name(
            canonical_name=canonical_name, credentials=context.credentials
        )

        if result.is_err():
            return SyftError(message=f"{result.err()}")

        return result.ok()

    @service_method(path="migration", name="register_migration_state")
    def register_migration_state(
        self,
        context: AuthedServiceContext,
        current_version: int,
        canonical_name: str,
    ) -> SyftObjectMigrationState | SyftError:
        obj = SyftObjectMigrationState(
            current_version=current_version, canonical_name=canonical_name
        )
        result = self.stash.set(migration_state=obj, credentials=context.credentials)

        if result.is_err():
            return SyftError(message=f"{result.err()}")

        return result.ok()

    def _find_klasses_pending_for_migration(
        self, context: AuthedServiceContext, object_types: list[type[SyftObject]]
    ) -> list[SyftObject]:
        klasses_to_be_migrated = []

        for object_type in object_types:
            canonical_name = object_type.__canonical_name__
            object_version = object_type.__version__

            migration_state = self.get_state(context, canonical_name)
            if isinstance(migration_state, SyftError):
                raise Exception(
                    f"Failed to get migration state for {canonical_name}. Error: {migration_state}"
                )
            if (
                migration_state is not None
                and migration_state.current_version != migration_state.latest_version
            ):
                klasses_to_be_migrated.append(object_type)
            else:
                self.register_migration_state(
                    context,
                    current_version=object_version,
                    canonical_name=canonical_name,
                )

        return klasses_to_be_migrated

    @service_method(
        path="migration.get_migration_objects",
        name="get_migration_objects",
        roles=ADMIN_ROLE_LEVEL,
    )
    def get_migration_objects(
        self,
        context: AuthedServiceContext,
        document_store_object_types: list[type[SyftObject]] | None = None,
    ) -> dict | SyftError:
        res = self._get_migration_objects(context, document_store_object_types)
        if res.is_err():
            return SyftError(message=res.value)
        else:
            return res.ok()

    def _get_migration_objects(
        self,
        context: AuthedServiceContext,
        document_store_object_types: list[type[SyftObject]] | None = None,
    ) -> Result[dict, str]:
        if document_store_object_types is None:
            document_store_object_types = [
                partition.settings.object_type
                for partition in self.store.partitions.values()
            ]

        klasses_to_migrate = self._find_klasses_pending_for_migration(
            context=context, object_types=document_store_object_types
        )

        if klasses_to_migrate:
            print(
                f"Classes in Document Store that need migration: {klasses_to_migrate}"
            )

        result = {}

        for klass in klasses_to_migrate:
            canonical_name = klass.__canonical_name__
            object_partition = self.store.partitions.get(canonical_name)
            if object_partition is None:
                continue
            objects_result = object_partition.all(
                context.credentials, has_permission=True
            )
            if objects_result.is_err():
                return objects_result
            objects = objects_result.ok()
            result[klass] = objects
        return Ok(result)

    @service_method(
        path="migration.update_migrated_objects",
        name="update_migrated_objects",
        roles=ADMIN_ROLE_LEVEL,
    )
    def update_migrated_objects(
        self, context: AuthedServiceContext, migrated_objects: list[SyftObject]
    ) -> SyftSuccess | SyftError:
        res = self._update_migrated_objects(context, migrated_objects)
        if res.is_err():
            return SyftError(message=res.value)
        else:
            return SyftSuccess(message=res.ok())

    def _update_migrated_objects(
        self, context: AuthedServiceContext, migrated_objects: list[SyftObject]
    ) -> Result[str, str]:
        for migrated_object in migrated_objects:
            klass = type(migrated_object)
            canonical_name = klass.__canonical_name__
            object_partition = self.store.partitions.get(canonical_name)
            qk = object_partition.settings.store_key.with_obj(migrated_object.id)
            result = object_partition._update(
                context.credentials,
                qk=qk,
                obj=migrated_object,
                has_permission=True,
                overwrite=True,
                allow_missing_keys=True,
            )

            if result.is_err():
                return result.err()
        return Ok(value="success")

    @service_method(
        path="migration.migrate_data",
        name="migrate_data",
        roles=ADMIN_ROLE_LEVEL,
    )
    def migrate_data(
        self,
        context: AuthedServiceContext,
        document_store_object_types: list[type[SyftObject]] | None = None,
    ) -> SyftSuccess | SyftError:
        # Track all object type that need migration for document store

        # get all objects, keyed by type (because we might want to have different rules for different types)
        # Q: will this be tricky with the protocol????
        # A: For now we will assume that the client will have the same version

        # Then, locally we write stuff that says
        # for klass, objects in migration_dict.items():
        # for object in objects:
        #   if isinstance(object, X):
        #        do something custom
        #   else:
        #       migrated_value = object.migrate_to(klass.__version__, context)
        #
        # migrated_values = [SyftObject]
        # client.migration.write_migrated_values(migrated_values)

        migration_objects_result = self._get_migration_objects(
            context, document_store_object_types
        )
        if migration_objects_result.is_err():
            return migration_objects_result
        migration_objects = migration_objects_result.ok()

        migrated_objects = []

        for klass, objects in migration_objects.items():
            canonical_name = klass.__canonical_name__
            # Migrate data for objects in document store
            print(f"Migrating data for: {canonical_name} table.")
            for object in objects:
                try:
                    migrated_value = object.migrate_to(klass.__version__, context)
                    migrated_objects.append(migrated_value)
                except Exception:
                    # stdlib
                    import traceback

                    print(traceback.format_exc())
                    return Err(
                        f"Failed to migrate data to {klass} for qk {klass.__version__}: {object.id}"
                    )

        objects_update_update_result = self._update_migrated_objects(
            context, migrated_objects
        )
        if objects_update_update_result.is_err():
            return SyftError(message=objects_update_update_result.value)

        # now action objects
        migration_actionobjects_result: dict[type[SyftObject], list[SyftObject]] = (
            self._get_migration_actionobjects(context)
        )

        if migration_actionobjects_result.is_err():
            return migration_actionobjects_result
        migration_actionobjects = migration_actionobjects_result.ok()

        migrated_actionobjects = []
        for klass, action_objects in migration_actionobjects.items():
            # these are Actions, ActionObjects, and possibly others
            for object in action_objects:
                try:
                    migrated_actionobject = object.migrate_to(
                        klass.__version__, context
                    )
                    migrated_actionobjects.append(migrated_actionobject)
                except Exception:
                    # stdlib
                    import traceback

                    print(traceback.format_exc())
                    return Err(
                        f"Failed to migrate data to {klass} for qk {klass.__version__}: {object.id}"
                    )

        actionobjects_update_update_result = self._update_migrated_actionobjects(
            context, migrated_actionobjects
        )
        if actionobjects_update_update_result.is_err():
            return SyftError(message=actionobjects_update_update_result.err())

        return SyftSuccess(message="Data upgraded to the latest version")

    @service_method(
        path="migration.get_migration_actionobjects",
        name="get_migration_actionobjects",
        roles=ADMIN_ROLE_LEVEL,
    )
    def get_migration_actionobjects(self, context: AuthedServiceContext):
        res = self._get_migration_actionobjects(context)
        if res.is_ok():
            return res.ok()
        else:
            return SyftError(message=res.value)

    def _get_migration_actionobjects(
        self, context: AuthedServiceContext
    ) -> Result[dict[type[SyftObject], SyftObject], str]:
        # Track all object types from action store
        action_object_types = [Action, ActionObject]
        action_object_types.extend(ActionObject.__subclasses__())

        action_object_pending_migration = self._find_klasses_pending_for_migration(
            context=context, object_types=action_object_types
        )
        result_dict = {x: [] for x in action_object_pending_migration}
        action_store = context.node.action_store
        action_store_objects_result = action_store._all(
            context.credentials, has_permission=True
        )
        if action_store_objects_result.is_err():
            return action_store_objects_result
        action_store_objects = action_store_objects_result.ok()

        for obj in action_store_objects:
            if type(obj) in result_dict:
                result_dict[type(obj)].append(obj)
        return Ok(result_dict)

    @service_method(
        path="migration.update_migrated_actionobjects",
        name="update_migrated_actionobjects",
        roles=ADMIN_ROLE_LEVEL,
    )
    def update_migrated_actionobjects(
        self, context: AuthedServiceContext, objects: list[SyftObject]
    ) -> SyftSuccess | SyftError:
        res = self._update_migrated_actionobjects(context, objects)
        if res.is_ok():
            return SyftSuccess(message="succesfully migrated actionobjects")
        else:
            return SyftError(message=res.value)

    def _update_migrated_actionobjects(
        self, context: AuthedServiceContext, objects: list[SyftObject]
    ) -> Result[str, str]:
        # Track all object types from action store
        action_store = context.node.action_store
        for obj in objects:
            res = action_store.set(
                uid=obj.id, credentials=context.credentials, syft_object=obj
            )
            if res.is_err():
                return res
        return Ok("success")
