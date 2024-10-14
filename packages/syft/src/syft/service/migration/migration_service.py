# stdlib
from collections import defaultdict
import logging
from typing import Any

# syft absolute
import syft

# relative
from ...serde.serializable import serializable
from ...store.db.db import DBManager
from ...store.db.stash import ObjectStash
from ...store.document_store_errors import NotFoundException
from ...types.blob_storage import BlobStorageEntry
from ...types.errors import SyftException
from ...types.result import as_result
from ...types.syft_object import SyftObject
from ...types.syft_object_registry import SyftObjectRegistry
from ...types.twin_object import TwinObject
from ...types.uid import UID
from ..action.action_object import Action
from ..action.action_object import ActionObject
from ..action.action_permissions import ActionObjectPermission
from ..action.action_permissions import StoragePermission
from ..action.action_store import ActionObjectStash
from ..context import AuthedServiceContext
from ..response import SyftError
from ..response import SyftSuccess
from ..service import AbstractService
from ..service import service_method
from ..sync.sync_service import get_store
from ..sync.sync_service import get_store_by_type
from ..user.user_roles import ADMIN_ROLE_LEVEL
from ..user.user_roles import DATA_SCIENTIST_ROLE_LEVEL
from ..worker.utils import DEFAULT_WORKER_POOL_NAME
from .object_migration_state import MigrationData
from .object_migration_state import StoreMetadata
from .object_migration_state import SyftMigrationStateStash
from .object_migration_state import SyftObjectMigrationState

logger = logging.getLogger(__name__)


@serializable(canonical_name="MigrationService", version=1)
class MigrationService(AbstractService):
    stash: SyftMigrationStateStash

    def __init__(self, store: DBManager) -> None:
        self.stash = SyftMigrationStateStash(store=store)

    @service_method(path="migration", name="get_version")
    def get_version(self, context: AuthedServiceContext, canonical_name: str) -> int:
        """Search for the metadata for an object."""

        migration_state = self.stash.get_by_name(
            canonical_name=canonical_name, credentials=context.credentials
        ).unwrap()

        if migration_state is None:
            raise SyftException(
                public_message=f"No migration state exists for canonical name: {canonical_name}"
            )

        return migration_state.current_version

    @service_method(path="migration", name="get_state")
    @as_result(SyftException, NotFoundException)
    def get_state(
        self, context: AuthedServiceContext, canonical_name: str
    ) -> SyftObjectMigrationState:
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
        return self.stash.set(obj=obj, credentials=context.credentials).unwrap()

    @as_result(SyftException, NotFoundException)
    def _find_klasses_pending_for_migration(
        self, context: AuthedServiceContext, object_types: list[type[SyftObject]]
    ) -> list[type[SyftObject]]:
        klasses_to_be_migrated = []

        for object_type in object_types:
            canonical_name = object_type.__canonical_name__
            object_version = object_type.__version__

            try:
                migration_state = self.get_state(context, canonical_name).unwrap(
                    public_message=f"Failed to get migration state for {canonical_name}."
                )
                if int(migration_state.current_version) != int(
                    migration_state.latest_version
                ):
                    klasses_to_be_migrated.append(object_type)
            except NotFoundException:
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
    ) -> dict[type[SyftObject], StoreMetadata]:
        return self._get_all_store_metadata(
            context,
            document_store_object_types=document_store_object_types,
        ).unwrap()

    @as_result(SyftException)
    def _get_all_store_metadata(
        self,
        context: AuthedServiceContext,
        document_store_object_types: list[type[SyftObject]] | None = None,
    ) -> dict[type[SyftObject], StoreMetadata]:
        # metadata = permissions + storage permissions
        stashes = context.server.services.stashes
        store_metadata = {}

        for klass, stash in stashes.items():
            if (
                document_store_object_types is not None
                and klass not in document_store_object_types
            ):
                continue
            store_metadata[klass] = StoreMetadata(
                object_type=klass,
                permissions=stash.get_all_permissions().unwrap(),
                storage_permissions=stash.get_all_storage_permissions().unwrap(),
            )

        return store_metadata

    @as_result(SyftException)
    def _update_store_metadata_for_klass(
        self, context: AuthedServiceContext, metadata: StoreMetadata
    ) -> None:
        stash = self._search_stash_for_klass(context, metadata.object_type).unwrap()
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

        stash.add_permissions(permissions, ignore_missing=True).unwrap()
        stash.add_storage_permissions(storage_permissions, ignore_missing=True).unwrap()

    @as_result(SyftException)
    def _update_store_metadata(
        self, context: AuthedServiceContext, store_metadata: dict[type, StoreMetadata]
    ) -> None:
        print("Updating store metadata")
        for metadata in store_metadata.values():
            self._update_store_metadata_for_klass(context, metadata).unwrap()

    @as_result(SyftException)
    def _get_migration_objects(
        self,
        context: AuthedServiceContext,
        document_store_object_types: list[type[SyftObject]] | None = None,
        get_all: bool = False,
    ) -> dict[type[SyftObject], list[SyftObject]]:
        if document_store_object_types is None:
            document_store_object_types = list(context.server.services.stashes.keys())

        if get_all:
            klasses_to_migrate = document_store_object_types
        else:
            klasses_to_migrate = self._find_klasses_pending_for_migration(
                context=context, object_types=document_store_object_types
            ).unwrap()

        result = defaultdict(list)

        for klass in klasses_to_migrate:
            stash_or_err = self._search_stash_for_klass(context, klass)
            if stash_or_err.is_err():
                continue
            stash = stash_or_err.unwrap()

            for object in stash._data:
                actual_klass = type(object)
                use_klass = (
                    klass
                    if actual_klass.__canonical_name__ == klass.__canonical_name__
                    else actual_klass
                )
                result[use_klass].append(object)

        return dict(result)

    @as_result(SyftException)
    def _search_stash_for_klass(
        self, context: AuthedServiceContext, klass: type[SyftObject]
    ) -> ObjectStash:
        if issubclass(klass, ActionObject | TwinObject | Action):
            return context.server.services.action.stash

        stashes: dict[str, ObjectStash] = {  # type: ignore
            t.__canonical_name__: stash
            for t, stash in context.server.services.stashes.items()
        }

        mro = klass.__mro__
        class_index = 0
        object_stash = None
        while len(mro) > class_index:
            try:
                canonical_name = mro[class_index].__canonical_name__
            except AttributeError:
                # Classes without cname dont have a stash
                break
            object_stash = stashes.get(canonical_name)
            if object_stash is not None:
                break
            class_index += 1
        if object_stash is None:
            raise SyftException(public_message=f"Object stash not found for {klass}")
        return object_stash

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
    ) -> SyftSuccess:
        self._create_migrated_objects(
            context, migrated_objects, ignore_existing=ignore_existing
        ).unwrap()
        return SyftSuccess(message="Created migration objects!")

    @as_result(SyftException)
    def _create_migrated_objects(
        self,
        context: AuthedServiceContext,
        migrated_objects: dict[type[SyftObject], list[SyftObject]],
        ignore_existing: bool = True,
        skip_check_type: bool = False,
    ) -> dict[type[SyftObject], list[SyftObject]]:
        created_objects: dict[type[SyftObject], list[SyftObject]] = {}

        for key, objects in migrated_objects.items():
            created_objects[key] = []
            for migrated_object in objects:
                stash = self._search_stash_for_klass(
                    context, type(migrated_object)
                ).unwrap()

                result = stash.set(
                    context.credentials,
                    obj=migrated_object,
                    skip_check_type=skip_check_type,
                )
                # Exception from the new Error Handling pattern, no need to change
                if result.is_err():
                    # TODO: subclass a DuplicationKeyError
                    if ignore_existing and (
                        "Duplication Key Error" in result.err()._private_message  # type: ignore
                        or "Duplication Key Error" in result.err().public_message  # type: ignore
                    ):
                        print(
                            f"{type(migrated_object)} #{migrated_object.id} already exists"
                        )
                        continue
                    else:
                        result.unwrap()  # this will raise the exception inside the wrapper
                created_objects[key].append(result.unwrap())
        return created_objects

    @as_result(SyftException)
    def _update_migrated_objects(
        self, context: AuthedServiceContext, migrated_objects: list[SyftObject]
    ) -> SyftSuccess:
        for migrated_object in migrated_objects:
            stash = self._search_stash_for_klass(
                context, type(migrated_object)
            ).unwrap()

            stash.update(
                context.credentials,
                obj=migrated_object,
            ).unwrap()

        return SyftSuccess(message="Updated migration objects!")

    @as_result(SyftException)
    def _migrate_objects(
        self,
        context: AuthedServiceContext,
        migration_objects: dict[type[SyftObject], list[SyftObject]],
    ) -> list[SyftObject]:
        migrated_objects = []

        for klass, objects in migration_objects.items():
            canonical_name = klass.__canonical_name__
            latest_version = SyftObjectRegistry.get_latest_version(canonical_name)

            # Migrate data for objects in document store
            logger.info(
                f"Migrating data for: {canonical_name} table to version {latest_version}"
            )
            for object in objects:
                try:
                    migrated_value = object.migrate_to(latest_version, context)
                    migrated_objects.append(migrated_value)
                except Exception:
                    raise SyftException(
                        public_message=f"Failed to migrate data to {klass} for qk {klass.__version__}: {object.id}"
                    )
        return migrated_objects

    @service_method(
        path="migration.migrate_data",
        name="migrate_data",
        roles=ADMIN_ROLE_LEVEL,
    )
    def migrate_data(
        self,
        context: AuthedServiceContext,
        document_store_object_types: list[type[SyftObject]] | None = None,
    ) -> SyftSuccess:
        migration_objects = self._get_migration_objects(
            context, document_store_object_types
        ).unwrap()
        migrated_objects = self._migrate_objects(context, migration_objects).unwrap()
        self._update_migrated_objects(context, migrated_objects).unwrap()

        migration_actionobjects = self._get_migration_actionobjects(context).unwrap()
        migrated_actionobjects = self._migrate_objects(
            context, migration_actionobjects
        ).unwrap()
        self._update_migrated_actionobjects(context, migrated_actionobjects).unwrap()

        return SyftSuccess(message="Data upgraded to the latest version")

    @service_method(
        path="migration.get_migration_actionobjects",
        name="get_migration_actionobjects",
        roles=ADMIN_ROLE_LEVEL,
    )
    def get_migration_actionobjects(
        self, context: AuthedServiceContext, get_all: bool = False
    ) -> dict:
        return self._get_migration_actionobjects(context, get_all=get_all).unwrap()

    @as_result(SyftException)
    def _get_migration_actionobjects(
        self, context: AuthedServiceContext, get_all: bool = False
    ) -> dict[type[SyftObject], list[SyftObject]]:
        # Track all object types from action store
        action_object_types = [Action, ActionObject, TwinObject]
        action_object_types.extend(ActionObject.__subclasses__())
        klass_by_canonical_name: dict[str, type[SyftObject]] = {
            klass.__canonical_name__: klass for klass in action_object_types
        }

        action_object_pending_migration = self._find_klasses_pending_for_migration(
            context=context, object_types=action_object_types
        ).unwrap()
        result_dict: dict[type[SyftObject], list[SyftObject]] = defaultdict(list)
        action_stash = context.server.services.action.stash
        action_store_objects = action_stash.get_all(context.credentials).unwrap()

        for obj in action_store_objects:
            if get_all or type(obj) in action_object_pending_migration:
                klass = klass_by_canonical_name.get(obj.__canonical_name__, type(obj))
                result_dict[klass].append(obj)  # type: ignore
        return dict(result_dict)

    @as_result(SyftException)
    def _update_migrated_actionobjects(
        self, context: AuthedServiceContext, objects: list[SyftObject]
    ) -> str:
        action_store: ActionObjectStash = context.server.services.action.stash
        for obj in objects:
            action_store.set_or_update(
                uid=obj.id,
                credentials=context.credentials,
                syft_object=obj,
            ).unwrap()
        return "success"

    @service_method(
        path="migration.get_migration_data",
        name="get_migration_data",
        roles=ADMIN_ROLE_LEVEL,
    )
    def get_migration_data(self, context: AuthedServiceContext) -> MigrationData:
        store_objects = self._get_migration_objects(context, get_all=True).unwrap()
        action_objects = self._get_migration_actionobjects(
            context, get_all=True
        ).unwrap()
        blob_storage_objects = store_objects.pop(BlobStorageEntry, [])
        store_metadata = self._get_all_store_metadata(context).unwrap()
        return MigrationData(
            server_uid=context.server.id,
            signing_key=context.server.signing_key,
            syft_version=syft.__version__,
            default_pool_name=DEFAULT_WORKER_POOL_NAME,
            store_objects=store_objects,
            metadata=store_metadata,
            action_objects=action_objects,
            blob_storage_objects=blob_storage_objects,
        )

    @service_method(
        path="migration.apply_migration_data",
        name="apply_migration_data",
        roles=ADMIN_ROLE_LEVEL,
        unwrap_on_success=False,
    )
    def apply_migration_data(
        self,
        context: AuthedServiceContext,
        migration_data: MigrationData,
    ) -> SyftSuccess:
        # NOTE blob storage is migrated via client,
        # it needs access to both source and destination blob storages.
        if len(migration_data.blobs):
            raise SyftException(
                public_message="Blob storage migration is not supported by this endpoint, "
                "please use 'client.load_migration_data' instead."
            )

        created_objects = self._create_migrated_objects(
            context, migration_data.store_objects, skip_check_type=True
        ).unwrap()

        # migrate + apply store objects
        migrated_objects = self._migrate_objects(
            context,
            created_objects,
        ).unwrap()
        self._update_migrated_objects(context, migrated_objects).unwrap()

        # migrate+apply action objects
        migrated_actionobjects = self._migrate_objects(
            context, migration_data.action_objects
        ).unwrap()
        self._update_migrated_actionobjects(context, migrated_actionobjects).unwrap()

        # apply metadata
        self._update_store_metadata(context, migration_data.metadata).unwrap()
        return SyftSuccess(message="Migration completed successfully")

    @service_method(
        path="migration.reset_and_restore",
        name="reset_and_restore",
        roles=ADMIN_ROLE_LEVEL,
        unwrap_on_success=False,
    )
    def reset_and_restore(
        self,
        context: AuthedServiceContext,
        migration_data: MigrationData,
    ) -> SyftSuccess | SyftError:
        try:
            root_verify_key = context.server.verify_key
            context.server.db.init_tables(reset=True)
            context.credentials = root_verify_key
            self.apply_migration_data(context, migration_data)
        except Exception as e:
            return SyftError.from_exception(
                context=context,
                exc=e,
                include_traceback=True,
            )

        return SyftSuccess(message="Database reset successfully.")

    @service_method(
        path="migration._get_object",
        name="_get_object",
        roles=DATA_SCIENTIST_ROLE_LEVEL,
    )
    def _get_object(
        self, context: AuthedServiceContext, uid: UID, object_type: type
    ) -> Any:
        return (
            get_store_by_type(context, object_type)
            .get_by_uid(credentials=context.credentials, uid=uid)
            .unwrap()
        )

    @service_method(
        path="migration._update_object",
        name="_update_object",
        roles=ADMIN_ROLE_LEVEL,
    )
    def _update_object(self, context: AuthedServiceContext, object: Any) -> Any:
        return (
            get_store(context, object)
            .update(credentials=context.credentials, obj=object)
            .unwrap()
        )
