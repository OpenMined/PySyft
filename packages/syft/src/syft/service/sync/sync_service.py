# stdlib
from collections import defaultdict
import logging
from typing import Any

# third party
from result import Err
from result import Ok
from result import Result

# relative
from ...client.api import ServerIdentity
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
from ..api.api import TwinAPIEndpoint
from ..api.api_service import APIService
from ..code.user_code import UserCodeStatusCollection
from ..context import AuthedServiceContext
from ..job.job_stash import Job
from ..response import SyftError
from ..response import SyftSuccess
from ..service import AbstractService
from ..service import TYPE_TO_SERVICE
from ..service import service_method
from ..user.user_roles import ADMIN_ROLE_LEVEL
from .sync_stash import SyncStash
from .sync_state import SyncState

logger = logging.getLogger(__name__)


def get_store(context: AuthedServiceContext, item: SyncableSyftObject) -> Any:
    if isinstance(item, ActionObject):
        service = context.server.get_service("actionservice")  # type: ignore
        return service.store  # type: ignore
    service = context.server.get_service(TYPE_TO_SERVICE[type(item)])  # type: ignore
    return service.stash.partition


@instrument
@serializable(canonical_name="SyncService", version=1)
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
        store_to = context.server.get_service("actionservice").store  # type: ignore
        for permission in new_permissions:
            if permission.permission == ActionPermission.READ:
                store_to.add_permission(permission)

        blob_id = action_object.syft_blob_storage_entry_id
        if blob_id:
            store_to_blob = context.server.get_service(
                "blobstorageservice"
            ).stash.partition  # type: ignore
            for permission in new_permissions:
                if permission.permission == ActionPermission.READ:
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
            x.syft_server_location = context.server.id  # type: ignore
            x.syft_client_verify_key = context.credentials
            if hasattr(x, "server_uid"):
                x.server_uid = context.server.id  # type: ignore

    def transform_item(
        self,
        context: AuthedServiceContext,
        item: SyncableSyftObject,
    ) -> SyftObject:
        if isinstance(item, UserCodeStatusCollection):
            identity = ServerIdentity.from_server(context.server)
            res = {}
            for key in item.status_dict.keys():
                # todo, check if they are actually only two servers
                res[identity] = item.status_dict[key]
            item.status_dict = res

        self.set_obj_ids(context, item)
        return item

    def get_stash_for_item(
        self, context: AuthedServiceContext, item: SyftObject
    ) -> BaseStash:
        services = list(context.server.service_path_map.values())  # type: ignore

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

        if isinstance(item, TwinAPIEndpoint):
            # we need the side effect of set function
            # to create an action object
            apiservice: APIService = context.server.get_service("apiservice")  # type: ignore

            res = apiservice.set(context=context, endpoint=item)
            if isinstance(res, SyftError):
                return Err(res.message)
            else:
                return Ok(item)

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
                message = f"Synced {len(items)} items"
                if len(ignored_batches) > 0:
                    message += f", ignored {len(ignored_batches)} batches"
                if len(unignored_batches) > 0:
                    message += f", unignored {len(unignored_batches)} batches"
                return SyftSuccess(message=message)

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
                # TODO fix error handling
                uid = item.id.id
                item_permissions = store._get_permissions_for_uid(uid)
                if not item_permissions.is_err():
                    permissions[uid] = item_permissions.ok()

                # TODO fix error handling for storage permissions
                item_storage_permissions = store._get_storage_permissions_for_uid(uid)
                if not item_storage_permissions.is_err():
                    storage_permissions[uid] = item_storage_permissions.ok()
        return permissions, storage_permissions

    def _get_all_items_for_jobs(
        self,
        context: AuthedServiceContext,
    ) -> Result[tuple[list[SyncableSyftObject], dict[UID, str]], str]:
        """
        Returns all Jobs, along with their Logs, ExecutionOutputs and ActionObjects
        """
        items_for_jobs = []
        errors = {}

        job_service = context.server.get_service("jobservice")
        jobs = job_service.get_all(context)
        if isinstance(jobs, SyftError):
            return Err(jobs.message)

        for job in jobs:
            job_items_result = self._get_job_batch(context, job)
            if job_items_result.is_err():
                logger.info(
                    f"Job {job.id} could not be added to SyncState: {job_items_result.err()}"
                )
                errors[job.id] = job_items_result.err()
                continue
            items_for_jobs.extend(job_items_result.ok())

        return Ok((items_for_jobs, errors))

    def _get_job_batch(
        self, context: AuthedServiceContext, job: Job
    ) -> Result[list[SyncableSyftObject], str]:
        job_batch = [job]

        log_service = context.server.get_service("logservice")
        log = log_service.get(context, job.log_id)
        if isinstance(log, SyftError):
            return Err(log.message)
        job_batch.append(log)

        output_service = context.server.get_service("outputservice")
        output = output_service.get_by_job_id(context, job.id)
        if isinstance(output, SyftError):
            return Err(output.message)

        if output is not None:
            job_batch.append(output)
            job_result_ids = set(output.output_id_list)
        else:
            job_result_ids = set()

        if isinstance(job.result, ActionObject):
            job_result_ids.add(job.result.id.id)

        action_service = context.server.get_service("actionservice")
        for result_id in job_result_ids:
            action_object = action_service.get(context, result_id)
            if action_object.is_err():
                return action_object
            job_batch.append(action_object.ok())

        return Ok(job_batch)

    def get_all_syncable_items(
        self, context: AuthedServiceContext
    ) -> Result[tuple[list[SyncableSyftObject], dict[UID, str]], str]:
        all_items: list[SyncableSyftObject] = []

        # NOTE Jobs are handled separately
        services_to_sync = [
            "requestservice",
            "usercodeservice",
            "usercodestatusservice",
            "apiservice",
        ]

        for service_name in services_to_sync:
            service = context.server.get_service(service_name)
            items = service.get_all(context)
            if isinstance(items, SyftError):
                return Err(items.message)
            all_items.extend(items)

        # Gather jobs, logs, outputs and action objects
        items_for_jobs = self._get_all_items_for_jobs(context)
        if items_for_jobs.is_err():
            return items_for_jobs
        items_for_jobs, errors = items_for_jobs.ok()
        all_items.extend(items_for_jobs)

        return Ok((all_items, errors))

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

        objects, errors = objects_res.ok()
        permissions, storage_permissions = self.get_permissions(context, objects)

        previous_state = self.stash.get_latest(context=context)
        if previous_state.is_err():
            return previous_state
        previous_state = previous_state.ok()

        if previous_state is not None:
            previous_state_link = LinkedObject.from_obj(
                obj=previous_state,
                service_type=SyncService,
                server_uid=context.server.id,  # type: ignore
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
            server_uid=context.server.id,  # type: ignore
            server_name=context.server.name,  # type: ignore
            server_side_type=context.server.server_side_type,  # type: ignore
            previous_state_link=previous_state_link,
            permissions=permissions,
            storage_permissions=storage_permissions,
            ignored_batches=ignored_batches,
            object_sync_dates=object_sync_dates,
            errors=errors,
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
