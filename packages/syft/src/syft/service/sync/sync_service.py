# stdlib
from collections import defaultdict
from typing import Any

# third party
from result import Ok
from result import Result

# relative
from ...client.api import NodeIdentity
from ...serde.serializable import serializable
from ...store.document_store import BaseStash
from ...store.document_store import DocumentStore
from ...store.linked_obj import LinkedObject
from ...types.datetime import DateTime
from ...types.syft_object import SyftObject
from ...types.syncable_object import SyncableSyftObject
from ...types.uid import UID
from ...util.telemetry import instrument
from ..action.action_object import ActionObject
from ..action.action_permissions import ActionObjectPermission
from ..action.action_permissions import ActionPermission
from ..action.action_permissions import StoragePermission
from ..code.user_code import UserCodeStatusCollection
from ..context import AuthedServiceContext
from ..job.job_stash import Job
from ..output.output_service import ExecutionOutput
from ..response import SyftError
from ..response import SyftSuccess
from ..service import AbstractService
from ..service import TYPE_TO_SERVICE
from ..service import service_method
from ..user.user_roles import ADMIN_ROLE_LEVEL
from .sync_stash import SyncStash
from .sync_state import SyncState


def get_store(context: AuthedServiceContext, item: SyncableSyftObject) -> Any:
    if isinstance(item, ActionObject):
        service = context.node.get_service("actionservice")  # type: ignore
        return service.store  # type: ignore
    service = context.node.get_service(TYPE_TO_SERVICE[type(item)])  # type: ignore
    return service.stash.partition


@instrument
@serializable()
class SyncService(AbstractService):
    store: DocumentStore
    stash: SyncStash

    def __init__(self, store: DocumentStore):
        self.store = store
        self.stash = SyncStash(store=store)

    def add_actionobject_read_permissions(
        self,
        context: AuthedServiceContext,
        action_object: ActionObject,
        new_permissions: list[ActionObjectPermission],
    ) -> None:
        blob_id = action_object.syft_blob_storage_entry_id

        store_to = context.node.get_service("actionservice").store  # type: ignore
        store_to_blob = context.node.get_service("blobstorageservice").stash.partition  # type: ignore

        for permission in new_permissions:
            if permission.permission == ActionPermission.READ:
                store_to.add_permission(permission)

                permission_blob = ActionObjectPermission(
                    uid=blob_id,
                    permission=permission.permission,
                    credentials=permission.credentials,
                )
                store_to_blob.add_permission(permission_blob)

    def set_obj_ids(self, context: AuthedServiceContext, x: Any) -> None:
        if hasattr(x, "__dict__") and isinstance(x, SyftObject):
            for val in x.__dict__.values():
                if isinstance(val, list | tuple):
                    for v in val:
                        self.set_obj_ids(context, v)
                elif isinstance(val, dict):
                    for v in val.values():
                        self.set_obj_ids(context, v)
                else:
                    self.set_obj_ids(context, val)
            x.syft_node_location = context.node.id  # type: ignore
            x.syft_client_verify_key = context.credentials
            if hasattr(x, "node_uid"):
                x.node_uid = context.node.id  # type: ignore

    def transform_item(
        self,
        context: AuthedServiceContext,
        item: SyncableSyftObject,
    ) -> SyftObject:
        if isinstance(item, UserCodeStatusCollection):
            identity = NodeIdentity.from_node(context.node)
            res = {}
            for key in item.status_dict.keys():
                # todo, check if they are actually only two nodes
                res[identity] = item.status_dict[key]
            item.status_dict = res

        self.set_obj_ids(context, item)
        return item

    def get_stash_for_item(
        self, context: AuthedServiceContext, item: SyftObject
    ) -> BaseStash:
        services = list(context.node.service_path_map.values())  # type: ignore

        all_stashes = {}
        for serv in services:
            if (_stash := getattr(serv, "stash", None)) is not None:
                all_stashes[_stash.object_type] = _stash

        stash = all_stashes.get(type(item), None)
        return stash

    def add_permissions_for_item(
        self,
        context: AuthedServiceContext,
        item: SyftObject,
        new_permissions: list[ActionObjectPermission],
    ) -> None:
        if isinstance(item, ActionObject):
            raise ValueError("ActionObject permissions should be added separately")
        else:
            store = get_store(context, item)  # type: ignore
            for permission in new_permissions:
                if permission.permission == ActionPermission.READ:
                    store.add_permission(permission)

    def add_storage_permissions_for_item(
        self,
        context: AuthedServiceContext,
        item: SyftObject,
        new_permissions: list[StoragePermission],
    ) -> None:
        store = get_store(context, item)
        store.add_storage_permissions(new_permissions)

    def set_object(
        self, context: AuthedServiceContext, item: SyncableSyftObject
    ) -> Result[SyftObject, str]:
        stash = self.get_stash_for_item(context, item)
        creds = context.credentials

        exists = stash.get_by_uid(context.credentials, item.id).ok() is not None
        if exists:
            res = stash.update(creds, item)
        else:
            # Storage permissions are added separately
            res = stash.set(
                creds,
                item,
                add_storage_permission=False,
            )

        return res

    @service_method(
        path="sync.sync_items",
        name="sync_items",
        roles=ADMIN_ROLE_LEVEL,
    )
    def sync_items(
        self,
        context: AuthedServiceContext,
        items: list[SyncableSyftObject],
        permissions: list[ActionObjectPermission],
        storage_permissions: list[StoragePermission],
        ignored_batches: dict[UID, int],
        unignored_batches: set[UID],
    ) -> SyftSuccess | SyftError:
        permissions_dict = defaultdict(list)
        for permission in permissions:
            permissions_dict[permission.uid].append(permission)

        storage_permissions_dict = defaultdict(list)
        for storage_permission in storage_permissions:
            storage_permissions_dict[storage_permission.uid].append(storage_permission)

        for item in items:
            new_permissions = permissions_dict[item.id.id]
            new_storage_permissions = storage_permissions_dict[item.id.id]
            if isinstance(item, ActionObject):
                self.add_actionobject_read_permissions(context, item, new_permissions)
                self.add_storage_permissions_for_item(
                    context, item, new_storage_permissions
                )
            else:
                item = self.transform_item(context, item)  # type: ignore[unreachable]
                res = self.set_object(context, item)

                if res.is_ok():
                    self.add_permissions_for_item(context, item, new_permissions)
                    self.add_storage_permissions_for_item(
                        context, item, new_storage_permissions
                    )
                else:
                    return SyftError(message=f"Failed to sync {res.err()}")

        res = self.build_current_state(
            context,
            new_items=items,
            new_ignored_batches=ignored_batches,
            new_unignored_batches=unignored_batches,
        )

        if res.is_err():
            return SyftError(message=res.message)
        else:
            new_state = res.ok()
            res = self.stash.set(context.credentials, new_state)
            if res.is_err():
                return SyftError(message=res.message)
            else:
                return SyftSuccess(message=f"Synced {len(items)} items")

    @service_method(
        path="sync.get_permissions",
        name="get_permissions",
        roles=ADMIN_ROLE_LEVEL,
    )
    def get_permissions(
        self,
        context: AuthedServiceContext,
        items: list[SyncableSyftObject],
    ) -> tuple[dict[UID, set[str]], dict[UID, set[str]]]:
        permissions = {}
        storage_permissions = {}

        for item in items:
            store = get_store(context, item)
            if store is not None:
                _id = item.id.id
                permissions[_id] = store.permissions[_id]
                storage_permissions[_id] = store.storage_permissions[_id]
        return permissions, storage_permissions

    def get_all_syncable_items(
        self, context: AuthedServiceContext
    ) -> Result[list[SyncableSyftObject], str]:
        all_items = []

        services_to_sync = [
            "requestservice",
            "usercodeservice",
            "jobservice",
            "logservice",
            "outputservice",
            "usercodestatusservice",
        ]

        for service_name in services_to_sync:
            service = context.node.get_service(service_name)
            items = service.get_all(context)
            if isinstance(items, SyftError):
                return items
            all_items.extend(items)

        # NOTE we only need action objects from outputs for now
        action_object_ids = set()
        for obj in all_items:
            if isinstance(obj, ExecutionOutput):
                action_object_ids |= set(obj.output_id_list)
            elif isinstance(obj, Job) and obj.result is not None:
                if isinstance(obj.result, ActionObject):
                    obj.result = obj.result.as_empty()
                action_object_ids.add(obj.result.id)

        for uid in action_object_ids:
            action_object = context.node.get_service("actionservice").get(
                context, uid, resolve_nested=False
            )  # type: ignore
            if action_object.is_err():
                return action_object
            all_items.append(action_object.ok())

        return Ok(all_items)

    def build_current_state(
        self,
        context: AuthedServiceContext,
        new_items: list[SyncableSyftObject] | None = None,
        new_ignored_batches: dict[UID, int] | None = None,
        new_unignored_batches: set[UID] | None = None,
    ) -> Result[SyncState, str]:
        new_items = new_items if new_items is not None else []
        new_ignored_batches = (
            new_ignored_batches if new_ignored_batches is not None else {}
        )
        unignored_batches: set[UID] = (
            new_unignored_batches if new_unignored_batches is not None else set()
        )
        objects_res = self.get_all_syncable_items(context)
        if objects_res.is_err():
            return objects_res
        else:
            objects = objects_res.ok()
        permissions, storage_permissions = self.get_permissions(context, objects)

        previous_state = self.stash.get_latest(context=context)
        if previous_state.is_err():
            return previous_state
        previous_state = previous_state.ok()

        if previous_state is not None:
            previous_state_link = LinkedObject.from_obj(
                obj=previous_state,
                service_type=SyncService,
                node_uid=context.node.id,  # type: ignore
            )
            previous_ignored_batches = previous_state.ignored_batches
        else:
            previous_state_link = None
            previous_ignored_batches = {}

        ignored_batches = {
            **previous_ignored_batches,
            **new_ignored_batches,
        }

        ignored_batches = {
            k: v for k, v in ignored_batches.items() if k not in unignored_batches
        }

        object_sync_dates = (
            previous_state.object_sync_dates.copy() if previous_state else {}
        )
        for obj in new_items:
            object_sync_dates[obj.id.id] = DateTime.now()

        new_state = SyncState(
            node_uid=context.node.id,  # type: ignore
            node_name=context.node.name,  # type: ignore
            node_side_type=context.node.node_side_type,  # type: ignore
            previous_state_link=previous_state_link,
            permissions=permissions,
            storage_permissions=storage_permissions,
            ignored_batches=ignored_batches,
            object_sync_dates=object_sync_dates,
        )

        new_state.add_objects(objects, context)

        return Ok(new_state)

    @service_method(
        path="sync._get_state",
        name="_get_state",
        roles=ADMIN_ROLE_LEVEL,
    )
    def _get_state(self, context: AuthedServiceContext) -> SyncState | SyftError:
        res = self.build_current_state(context)
        if res.is_err():
            return SyftError(message=res.value)
        else:
            return res.ok()
