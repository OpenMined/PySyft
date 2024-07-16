# stdlib
from collections import defaultdict
import sys
from typing import cast

# third party
from result import Err
from result import Ok
from result import Result

# relative
from ...serde.serializable import serializable
from ...store.document_store import DocumentStore
from ...store.document_store import StorePartition
from ...types.blob_storage import BlobStorageEntry
from ...types.syft_object import SyftObject
from ..action.action_object import Action
from ..action.action_object import ActionObject
from ..action.action_permissions import ActionObjectPermission
from ..action.action_permissions import StoragePermission
from ..action.action_store import KeyValueActionStore
from ..context import AuthedServiceContext
from ..response import SyftError
from ..response import SyftSuccess
from ..service import AbstractService
from ..service import service_method
from ..user.user_roles import ADMIN_ROLE_LEVEL
from .object_migration_state import MigrationData
from .object_migration_state import StoreMetadata
from .object_migration_state import SyftMigrationStateStash
from .object_migration_state import SyftObjectMigrationState


@serializable(canonical_name="MigrationService", version=1)
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
    ) -> list[type[SyftObject]]:
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
        path="migration.get_all_store_metadata",
        name="get_all_store_metadata",
        roles=ADMIN_ROLE_LEVEL,
    )
    def get_all_store_metadata(
        self,
        context: AuthedServiceContext,
        document_store_object_types: list[type[SyftObject]] | None = None,
        include_action_store: bool = True,
    ) -> dict[str, StoreMetadata] | SyftError:
        res = self._get_all_store_metadata(
            context,
            document_store_object_types=document_store_object_types,
            include_action_store=include_action_store,
        )
        if res.is_err():
            return SyftError(message=res.value)
        else:
            return res.ok()

    def _get_partition_from_type(
        self,
        context: AuthedServiceContext,
        object_type: type[SyftObject],
    ) -> Result[KeyValueActionStore | StorePartition, str]:
        object_partition: KeyValueActionStore | StorePartition | None = None
        if issubclass(object_type, ActionObject):
            object_partition = cast(KeyValueActionStore, context.server.action_store)
        else:
            canonical_name = object_type.__canonical_name__  # type: ignore[unreachable]
            object_partition = self.store.partitions.get(canonical_name)

        if object_partition is None:
            return Err(f"Object partition not found for {object_type}")  # type: ignore

        return Ok(object_partition)

    def _get_store_metadata(
        self,
        context: AuthedServiceContext,
        object_type: type[SyftObject],
    ) -> Result[StoreMetadata, str]:
        object_partition = self._get_partition_from_type(context, object_type)
        if object_partition.is_err():
            return object_partition
        object_partition = object_partition.ok()

        permissions = object_partition.get_all_permissions()

        if permissions.is_err():
            return permissions
        permissions = permissions.ok()

        storage_permissions = object_partition.get_all_storage_permissions()
        if storage_permissions.is_err():
            return storage_permissions
        storage_permissions = storage_permissions.ok()

        return Ok(
            StoreMetadata(
                object_type=object_type,
                permissions=permissions,
                storage_permissions=storage_permissions,
            )
        )

    def _get_all_store_metadata(
        self,
        context: AuthedServiceContext,
        document_store_object_types: list[type[SyftObject]] | None = None,
        include_action_store: bool = True,
    ) -> Result[dict[str, list[str]], str]:
        if document_store_object_types is None:
            document_store_object_types = self.store.get_partition_object_types()

        store_metadata = {}
        for klass in document_store_object_types:
            result = self._get_store_metadata(context, klass)
            if result.is_err():
                return result
            store_metadata[klass] = result.ok()

        if include_action_store:
            result = self._get_store_metadata(context, ActionObject)
            if result.is_err():
                return result
            store_metadata[ActionObject] = result.ok()

        return Ok(store_metadata)

    @service_method(
        path="migration.update_store_metadata",
        name="update_store_metadata",
        roles=ADMIN_ROLE_LEVEL,
    )
    def update_store_metadata(
        self, context: AuthedServiceContext, store_metadata: dict[type, StoreMetadata]
    ) -> SyftSuccess | SyftError:
        res = self._update_store_metadata(context, store_metadata)
        if res.is_err():
            return SyftError(message=res.value)
        else:
            return SyftSuccess(message=res.ok())

    def _update_store_metadata_for_klass(
        self, context: AuthedServiceContext, metadata: StoreMetadata
    ) -> Result[str, str]:
        object_partition = self._get_partition_from_type(context, metadata.object_type)
        if object_partition.is_err():
            return object_partition
        object_partition = object_partition.ok()

        permissions = [
            ActionObjectPermission.from_permission_string(uid, perm_str)
            for uid, perm_strs in metadata.permissions.items()
            for perm_str in perm_strs
        ]

        storage_permissions = [
            StoragePermission(uid, server_uid)
            for uid, server_uids in metadata.storage_permissions.items()
            for server_uid in server_uids
        ]

        object_partition.add_permissions(permissions)
        object_partition.add_storage_permissions(storage_permissions)

        return Ok("success")

    def _update_store_metadata(
        self, context: AuthedServiceContext, store_metadata: dict[type, StoreMetadata]
    ) -> Result[str, str]:
        print("Updating store metadata")
        for metadata in store_metadata.values():
            result = self._update_store_metadata_for_klass(context, metadata)
            if result.is_err():
                return result
        return Ok("success")

    @service_method(
        path="migration.get_migration_objects",
        name="get_migration_objects",
        roles=ADMIN_ROLE_LEVEL,
    )
    def get_migration_objects(
        self,
        context: AuthedServiceContext,
        document_store_object_types: list[type[SyftObject]] | None = None,
        get_all: bool = False,
    ) -> dict | SyftError:
        res = self._get_migration_objects(context, document_store_object_types, get_all)
        if res.is_err():
            return SyftError(message=res.value)
        else:
            return res.ok()

    def _get_migration_objects(
        self,
        context: AuthedServiceContext,
        document_store_object_types: list[type[SyftObject]] | None = None,
        get_all: bool = False,
    ) -> Result[dict[type[SyftObject], list[SyftObject]], str]:
        if document_store_object_types is None:
            document_store_object_types = self.store.get_partition_object_types()

        if get_all:
            klasses_to_migrate = document_store_object_types
        else:
            klasses_to_migrate = self._find_klasses_pending_for_migration(
                context=context, object_types=document_store_object_types
            )

        result = defaultdict(list)

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
            for object in objects:
                actual_klass = type(object)
                use_klass = (
                    klass
                    if actual_klass.__canonical_name__ == klass.__canonical_name__
                    else actual_klass
                )
                result[use_klass].append(object)

        return Ok(dict(result))

    def _search_partition_for_object(
        self, context: AuthedServiceContext, obj: SyftObject
    ) -> Result[StorePartition, str]:
        klass = type(obj)
        mro = klass.__mro__
        class_index = 0
        object_partition = None
        while len(mro) > class_index:
            canonical_name = mro[class_index].__canonical_name__
            object_partition = self.store.partitions.get(canonical_name)
            if object_partition is not None:
                break
            class_index += 1
        if object_partition is None:
            return Err(f"Object partition not found for {klass}")
        return Ok(object_partition)

    @service_method(
        path="migration.create_migrated_objects",
        name="create_migrated_objects",
        roles=ADMIN_ROLE_LEVEL,
    )
    def create_migrated_objects(
        self,
        context: AuthedServiceContext,
        migrated_objects: list[SyftObject],
        ignore_existing: bool = True,
    ) -> SyftSuccess | SyftError:
        res = self._create_migrated_objects(context, migrated_objects)
        if res.is_err():
            return SyftError(message=res.value)
        else:
            return SyftSuccess(message=res.ok())

    def _create_migrated_objects(
        self,
        context: AuthedServiceContext,
        migrated_objects: list[SyftObject],
        ignore_existing: bool = True,
    ) -> Result[str, str]:
        for migrated_object in migrated_objects:
            object_partition_or_err = self._search_partition_for_object(
                context, migrated_object
            )
            if object_partition_or_err.is_err():
                return object_partition_or_err
            object_partition = object_partition_or_err.ok()

            # upsert the object
            result = object_partition.set(
                context.credentials,
                obj=migrated_object,
            )
            if result.is_err():
                if ignore_existing and "Duplication Key Error" in result.value:
                    print(
                        f"{type(migrated_object)} #{migrated_object.id} already exists"
                    )
                    continue
                else:
                    return result

        return Ok(value="success")

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
            object_partition_or_err = self._search_partition_for_object(
                context, migrated_object
            )
            if object_partition_or_err.is_err():
                return object_partition_or_err
            object_partition = object_partition_or_err.ok()

            # canonical_name = mro[class_index].__canonical_name__
            # object_partition = self.store.partitions.get(canonical_name)

            # print(klass, canonical_name, object_partition)
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
                print("ERR:", result.value, file=sys.stderr)
                print("ERR:", type(migrated_object), file=sys.stderr)
                print("ERR:", migrated_object, file=sys.stderr)
                # return result
        return Ok(value="success")

    def _migrate_objects(
        self,
        context: AuthedServiceContext,
        migration_objects: dict[type[SyftObject], list[SyftObject]],
    ) -> Result[list[SyftObject], str]:
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
        return Ok(migrated_objects)

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

        migrated_objects_result = self._migrate_objects(context, migration_objects)
        if migrated_objects_result.is_err():
            return SyftError(message=migrated_objects_result.err())
        migrated_objects = migrated_objects_result.ok()

        objects_update_update_result = self._update_migrated_objects(
            context, migrated_objects
        )
        if objects_update_update_result.is_err():
            return SyftError(message=objects_update_update_result.value)

        migration_actionobjects_result = self._get_migration_actionobjects(context)

        if migration_actionobjects_result.is_err():
            return SyftError(message=migration_actionobjects_result.err())
        migration_actionobjects = migration_actionobjects_result.ok()

        migrated_actionobjects = self._migrate_objects(context, migration_actionobjects)
        if migrated_actionobjects.is_err():
            return SyftError(message=migrated_actionobjects.err())
        migrated_actionobjects = migrated_actionobjects.ok()

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
    def get_migration_actionobjects(
        self, context: AuthedServiceContext, get_all: bool = False
    ) -> dict | SyftError:
        res = self._get_migration_actionobjects(context, get_all=get_all)
        if res.is_ok():
            return res.ok()
        else:
            return SyftError(message=res.value)

    def _get_migration_actionobjects(
        self, context: AuthedServiceContext, get_all: bool = False
    ) -> Result[dict[type[SyftObject], list[SyftObject]], str]:
        # Track all object types from action store
        action_object_types = [Action, ActionObject]
        action_object_types.extend(ActionObject.__subclasses__())
        klass_by_canonical_name: dict[str, type[SyftObject]] = {
            klass.__canonical_name__: klass for klass in action_object_types
        }

        action_object_pending_migration = self._find_klasses_pending_for_migration(
            context=context, object_types=action_object_types
        )
        result_dict: dict[type[SyftObject], list[SyftObject]] = defaultdict(list)
        action_store = context.server.action_store
        action_store_objects_result = action_store._all(
            context.credentials, has_permission=True
        )
        if action_store_objects_result.is_err():
            return action_store_objects_result
        action_store_objects = action_store_objects_result.ok()

        for obj in action_store_objects:
            if get_all or type(obj) in action_object_pending_migration:
                klass = klass_by_canonical_name.get(obj.__canonical_name__, type(obj))
                result_dict[klass].append(obj)  # type: ignore
        return Ok(dict(result_dict))

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
        action_store = context.server.action_store
        for obj in objects:
            res = action_store.set(
                uid=obj.id, credentials=context.credentials, syft_object=obj
            )
            if res.is_err():
                return res
        return Ok("success")

    @service_method(
        path="migration.get_migration_data",
        name="get_migration_data",
        roles=ADMIN_ROLE_LEVEL,
    )
    def get_migration_data(
        self, context: AuthedServiceContext
    ) -> MigrationData | SyftError:
        store_objects_result = self._get_migration_objects(context, get_all=True)
        if store_objects_result.is_err():
            return SyftError(message=store_objects_result.err())
        store_objects = store_objects_result.ok()

        action_objects_result = self._get_migration_actionobjects(context, get_all=True)
        if action_objects_result.is_err():
            return SyftError(message=action_objects_result.err())
        action_objects = action_objects_result.ok()

        blob_storage_objects = store_objects.pop(BlobStorageEntry, [])

        store_metadata_result = self._get_all_store_metadata(context)
        if store_metadata_result.is_err():
            return SyftError(message=store_metadata_result.err())
        store_metadata = store_metadata_result.ok()

        return MigrationData(
            server_uid=context.server.id,
            signing_key=context.server.signing_key,
            store_objects=store_objects,
            metadata=store_metadata,
            action_objects=action_objects,
            blob_storage_objects=blob_storage_objects,
        )

    @service_method(
        path="migration.apply_migration_data",
        name="apply_migration_data",
        roles=ADMIN_ROLE_LEVEL,
    )
    def apply_migration_data(
        self,
        context: AuthedServiceContext,
        migration_data: MigrationData,
    ) -> SyftSuccess | SyftError:
        # NOTE blob storage is migrated via client,
        # it needs access to both source and destination blob storages.
        if len(migration_data.blobs):
            return SyftError(
                message="Blob storage migration is not supported by this endpoint, "
                "please use 'client.load_migration_data' instead."
            )

        # migrate + apply store objects
        migrated_objects_result = self._migrate_objects(
            context, migration_data.store_objects
        )
        if migrated_objects_result.is_err():
            return SyftError(message=migrated_objects_result.err())
        migrated_objects = migrated_objects_result.ok()
        store_objects_result = self._create_migrated_objects(context, migrated_objects)
        if store_objects_result.is_err():
            return SyftError(message=store_objects_result.err())

        # migrate+apply action objects
        migrated_actionobjects = self._migrate_objects(
            context, migration_data.action_objects
        )
        if migrated_actionobjects.is_err():
            return SyftError(message=migrated_actionobjects.err())
        migrated_actionobjects = migrated_actionobjects.ok()
        action_objects_result = self._update_migrated_actionobjects(
            context, migrated_actionobjects
        )
        if action_objects_result.is_err():
            return SyftError(message=action_objects_result.err())

        # apply metadata
        metadata_result = self._update_store_metadata(context, migration_data.metadata)
        if metadata_result.is_err():
            return SyftError(message=metadata_result.err())

        return SyftSuccess(message="Migration completed successfully")
