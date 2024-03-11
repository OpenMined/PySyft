# stdlib
from collections import defaultdict
from typing import Any
from typing import cast

# third party
from result import Result

# relative
from ...abstract_node import AbstractNode
from ...client.api import NodeIdentity
from ...node.credentials import SyftVerifyKey
from ...serde.serializable import serializable
from ...store.document_store import BaseStash
from ...store.document_store import DocumentStore
from ...store.linked_obj import LinkedObject
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
        permissions_other: list[str],
    ) -> None:
        read_permissions = [x for x in permissions_other if "READ" in x]

        _id = action_object.id.id
        blob_id = action_object.syft_blob_storage_entry_id

        store_to = context.node.get_service("actionservice").store  # type: ignore
        store_to_blob = context.node.get_service("blobstorageservice").stash.partition  # type: ignore

        for read_permission in read_permissions:
            creds, perm_str = read_permission.split("_")
            perm = ActionPermission[perm_str]
            permission = ActionObjectPermission(
                uid=_id, permission=perm, credentials=SyftVerifyKey(creds)
            )
            store_to.add_permission(permission)

            permission_blob = ActionObjectPermission(
                uid=blob_id, permission=perm, credentials=SyftVerifyKey(creds)
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
        self, context: AuthedServiceContext, item: SyftObject
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
        permissions_other: set[ActionObjectPermission],
    ) -> None:
        if isinstance(item, Job) and context.node.node_side_type.value == "low":  # type: ignore
            _id = item.id
            read_permissions = [x for x in permissions_other if "READ" in x]  # type: ignore
            job_store = context.node.get_service("jobservice").stash.partition  # type: ignore
            for read_permission in read_permissions:
                creds, perm_str = read_permission.split("_")
                perm = ActionPermission[perm_str]
                permission = ActionObjectPermission(
                    uid=_id, permission=perm, credentials=SyftVerifyKey(creds)
                )
                job_store.add_permission(permission)

    def add_storage_permissions_for_item(
        self,
        context: AuthedServiceContext,
        item: SyftObject,
        permissions_other: set[UID],
    ) -> None:
        _id = item.id.id
        permissions = [
            StoragePermission(uid=_id, node_uid=p) for p in permissions_other
        ]

        store = get_store(context, item)
        store.add_storage_permissions(permissions)

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
        items: list[ActionObject | SyftObject],
        permissions: dict[UID, set[str]],
        storage_permissions: dict[UID, set[UID]],
    ) -> SyftSuccess | SyftError:
        permissions = defaultdict(set, permissions)
        storage_permissions = defaultdict(set, storage_permissions)
        for item in items:
            new_permissions = permissions[item.id.id]
            new_storage_permissions = storage_permissions[item.id.id]
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

    @service_method(
        path="sync._get_state",
        name="_get_state",
        roles=ADMIN_ROLE_LEVEL,
    )
    def _get_state(
        self, context: AuthedServiceContext, add_to_store: bool = False
    ) -> SyncState | SyftError:
        node = cast(AbstractNode, context.node)

        new_state = SyncState(node_uid=node.id)

        services_to_sync = [
            "requestservice",
            "usercodeservice",
            "jobservice",
            "logservice",
            "outputservice",
            "usercodestatusservice",
        ]

        for service_name in services_to_sync:
            service = node.get_service(service_name)
            items = service.get_all(context)
            new_state.add_objects(items, api=node.root_client.api)  # type: ignore

        # TODO workaround, we only need action objects from outputs for now
        action_object_ids = set()
        for obj in new_state.objects.values():
            if isinstance(obj, ExecutionOutput):
                action_object_ids |= set(obj.output_id_list)
            elif isinstance(obj, Job) and obj.result is not None:
                if isinstance(obj.result, ActionObject):
                    obj.result = obj.result.as_empty()
                action_object_ids.add(obj.result.id)

        action_objects = []
        for uid in action_object_ids:
            action_object = node.get_service("actionservice").get(context, uid)  # type: ignore
            if action_object.is_err():
                return SyftError(message=action_object.err())
            action_objects.append(action_object.ok())
        new_state.add_objects(action_objects)

        new_state._build_dependencies(api=node.root_client.api)  # type: ignore

        permissions, storage_permissions = self.get_permissions(
            context, new_state.objects.values()
        )
        new_state.permissions = permissions
        new_state.storage_permissions = storage_permissions

        previous_state = self.stash.get_latest(context=context)
        if previous_state is not None:
            new_state.previous_state_link = LinkedObject.from_obj(
                obj=previous_state,
                service_type=SyncService,
                node_uid=context.node.id,  # type: ignore
            )

        if add_to_store:
            self.stash.set(context.credentials, new_state)

        return new_state
