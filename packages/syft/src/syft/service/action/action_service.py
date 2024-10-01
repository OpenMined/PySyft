# stdlib
import importlib
import logging
from typing import Any

# third party
import numpy as np

# relative
from ...serde.serializable import serializable
from ...server.credentials import SyftVerifyKey
from ...store.db.db import DBManager
from ...store.document_store_errors import NotFoundException
from ...store.document_store_errors import StashException
from ...types.datetime import DateTime
from ...types.errors import SyftException
from ...types.result import as_result
from ...types.syft_object import SyftObject
from ...types.twin_object import TwinObject
from ...types.uid import UID
from ..code.user_code import UserCode
from ..code.user_code import execute_byte_code
from ..context import AuthedServiceContext
from ..policy.policy import OutputPolicy
from ..policy.policy import retrieve_from_db
from ..response import SyftResponseMessage
from ..response import SyftSuccess
from ..response import SyftWarning
from ..service import AbstractService
from ..service import SERVICE_TO_TYPES
from ..service import TYPE_TO_SERVICE
from ..service import UserLibConfigRegistry
from ..service import service_method
from ..user.user_roles import ADMIN_ROLE_LEVEL
from ..user.user_roles import GUEST_ROLE_LEVEL
from ..user.user_roles import ServiceRole
from .action_endpoint import CustomEndpointActionObject
from .action_object import Action
from .action_object import ActionObject
from .action_object import ActionObjectPointer
from .action_object import ActionType
from .action_object import AnyActionObject
from .action_object import TwinMode
from .action_permissions import ActionObjectPermission
from .action_permissions import ActionObjectREAD
from .action_permissions import ActionPermission
from .action_permissions import StoragePermission
from .action_store import ActionObjectStash
from .action_types import action_type_for_type
from .numpy import NumpyArrayObject
from .pandas import PandasDataFrameObject  # noqa: F401
from .pandas import PandasSeriesObject  # noqa: F401

logger = logging.getLogger(__name__)


@serializable(canonical_name="ActionService", version=1)
class ActionService(AbstractService):
    stash: ActionObjectStash

    def __init__(self, store: DBManager) -> None:
        self.stash = ActionObjectStash(store)

    @service_method(path="action.np_array", name="np_array")
    def np_array(self, context: AuthedServiceContext, data: Any) -> Any:
        # TODO: REMOVE!
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        # cast here since we are sure that AuthedServiceContext has a server

        np_obj = NumpyArrayObject(
            dtype=data.dtype,
            shape=data.shape,
            syft_action_data_cache=data,
            syft_server_location=context.server.id,
            syft_client_verify_key=context.credentials,
        )
        blob_store_result = np_obj._save_to_blob_storage().unwrap()
        if isinstance(blob_store_result, SyftWarning):
            logger.debug(blob_store_result.message)

        np_pointer = self._set(context, np_obj)
        return np_pointer

    @service_method(
        path="action.set",
        name="set",
        roles=GUEST_ROLE_LEVEL,
    )
    def set(
        self,
        context: AuthedServiceContext,
        action_object: ActionObject | TwinObject,
        add_storage_permission: bool = True,
        ignore_detached_objs: bool = False,
    ) -> ActionObject:
        return self._set(
            context,
            action_object,
            has_result_read_permission=True,
            add_storage_permission=add_storage_permission,
            ignore_detached_objs=ignore_detached_objs,
        ).unwrap()

    def is_detached_obj(
        self,
        action_object: ActionObject | TwinObject,
        ignore_detached_obj: bool = False,
    ) -> bool:
        """
        A detached object is an object that is not yet saved to the blob storage.
        """
        if (
            isinstance(action_object, TwinObject)
            and (
                (
                    action_object.mock_obj.syft_action_saved_to_blob_store
                    and action_object.mock_obj.syft_blob_storage_entry_id is None
                )
                or (
                    action_object.private_obj.syft_action_saved_to_blob_store
                    and action_object.private_obj.syft_blob_storage_entry_id is None
                )
            )
            and not ignore_detached_obj
        ):
            return True
        if isinstance(action_object, ActionObject) and (
            action_object.syft_action_saved_to_blob_store
            and action_object.syft_blob_storage_entry_id is None
            and not ignore_detached_obj
        ):
            return True
        return False

    @as_result(StashException, SyftException)
    def _set(
        self,
        context: AuthedServiceContext,
        action_object: ActionObject | TwinObject,
        has_result_read_permission: bool = False,
        add_storage_permission: bool = True,
        ignore_detached_objs: bool = False,
    ) -> ActionObject:
        if self.is_detached_obj(action_object, ignore_detached_objs):
            raise SyftException(
                public_message="You uploaded an ActionObject that is not yet in the blob storage"
            )

        """Save an object to the action store"""
        # 🟡 TODO 9: Create some kind of type checking / protocol for SyftSerializable

        if isinstance(action_object, ActionObject):
            action_object.syft_created_at = DateTime.now()
            (
                action_object._clear_cache()
                if action_object.syft_action_saved_to_blob_store
                else None
            )
        else:  # TwinObject
            action_object.private_obj.syft_created_at = DateTime.now()  # type: ignore[unreachable]
            action_object.mock_obj.syft_created_at = DateTime.now()

            # Clear cache if data is saved to blob storage
            (
                action_object.private_obj._clear_cache()
                if action_object.private_obj.syft_action_saved_to_blob_store
                else None
            )
            (
                action_object.mock_obj._clear_cache()
                if action_object.mock_obj.syft_action_saved_to_blob_store
                else None
            )

        # If either context or argument is True, has_result_read_permission is True
        has_result_read_permission = (
            context.extra_kwargs.get("has_result_read_permission", False)
            or has_result_read_permission
        )

        self.stash.set_or_update(
            uid=action_object.id,
            credentials=context.credentials,
            syft_object=action_object,
            has_result_read_permission=has_result_read_permission,
            add_storage_permission=add_storage_permission,
        ).unwrap()

        if isinstance(action_object, TwinObject):
            # give read permission to the mock
            # if mock is saved to blob store, then add READ permission
            if action_object.mock_obj.syft_action_saved_to_blob_store:
                blob_id = action_object.mock_obj.syft_blob_storage_entry_id
                permission = ActionObjectPermission(blob_id, ActionPermission.ALL_READ)
                # add_permission is not resultified.
                context.server.services.blob_storage.stash.add_permission(permission)

            if has_result_read_permission:
                action_object = action_object.private
            else:
                action_object = action_object.mock

        action_object.syft_point_to(context.server.id)

        return action_object

    @service_method(
        path="action.is_resolved", name="is_resolved", roles=GUEST_ROLE_LEVEL
    )
    def is_resolved(
        self,
        context: AuthedServiceContext,
        uid: UID,
    ) -> bool:
        """Get an object from the action store"""
        obj = self._get(context, uid).unwrap()

        if obj.is_link:
            result = self.resolve_links(
                context, obj.syft_action_data.action_object_id.id
            ).unwrap()
            return result.syft_resolved

        # If it's a leaf but not resolved yet, return false
        if not obj.syft_resolved:
            return False

        # If it's not an action data link or non resolved (empty). It's resolved
        return True

    @as_result(StashException, NotFoundException)
    def resolve_links(
        self,
        context: AuthedServiceContext,
        uid: UID,
        twin_mode: TwinMode = TwinMode.PRIVATE,
    ) -> ActionObject:
        """Get an object from the action store"""
        # If user has permission to get the object / object exists
        result = self.stash.get(uid=uid, credentials=context.credentials).unwrap()

        # If it's not a leaf
        if result.is_link:
            return self.resolve_links(
                context, result.syft_action_data.action_object_id.id, twin_mode
            ).unwrap()

        # If it's a leaf
        return result

    @service_method(path="action.get", name="get", roles=GUEST_ROLE_LEVEL)
    def get(
        self,
        context: AuthedServiceContext,
        uid: UID,
        twin_mode: TwinMode = TwinMode.PRIVATE,
        resolve_nested: bool = True,
    ) -> ActionObject | TwinObject:
        """Get an object from the action store"""
        return self._get(
            context, uid, twin_mode, resolve_nested=resolve_nested
        ).unwrap()

    @as_result(StashException, NotFoundException, SyftException)
    def _get(
        self,
        context: AuthedServiceContext,
        uid: UID,
        twin_mode: TwinMode = TwinMode.PRIVATE,
        has_permission: bool = False,
        resolve_nested: bool = True,
    ) -> ActionObject | TwinObject:
        """Get an object from the action store"""
        obj = self.stash.get(
            uid=uid, credentials=context.credentials, has_permission=has_permission
        ).unwrap()

        # TODO: Is this necessary?
        if context.server is None:
            raise SyftException(public_message=f"Server not found. Context: {context}")
        obj._set_obj_location_(
            context.server.id,
            context.credentials,
        )

        # Resolve graph links
        if not isinstance(obj, TwinObject) and resolve_nested and obj.is_link:  # type: ignore [unreachable]
            # if not self.is_resolved(  # type: ignore [unreachable]
            #     context, obj.syft_action_data.action_object_id.id
            # ):
            #     raise SyftException(public_message="This object is not resolved yet.")

            return self.resolve_links(  # type: ignore
                context, obj.syft_action_data.action_object_id.id, twin_mode
            ).unwrap()

        if isinstance(obj, TwinObject):
            if twin_mode == TwinMode.PRIVATE:
                obj = obj.private
                obj.syft_point_to(context.server.id)
            elif twin_mode == TwinMode.MOCK:
                obj = obj.mock
                obj.syft_point_to(context.server.id)
            else:
                obj.mock.syft_point_to(context.server.id)
                obj.private.syft_point_to(context.server.id)

        return obj

    @service_method(
        path="action.get_pointer", name="get_pointer", roles=GUEST_ROLE_LEVEL
    )
    def get_pointer(
        self, context: AuthedServiceContext, uid: UID
    ) -> ActionObjectPointer:
        """Get a pointer from the action store"""
        obj = self.stash.get_pointer(
            uid=uid, credentials=context.credentials, server_uid=context.server.id
        ).unwrap()

        obj._set_obj_location_(
            context.server.id,
            context.credentials,
        )

        return obj

    @service_method(path="action.get_mock", name="get_mock", roles=GUEST_ROLE_LEVEL)
    def get_mock(self, context: AuthedServiceContext, uid: UID) -> SyftObject:
        """Get a pointer from the action store"""
        return self.stash.get_mock(credentials=context.credentials, uid=uid).unwrap()

    @service_method(
        path="action.has_storage_permission",
        name="has_storage_permission",
        roles=GUEST_ROLE_LEVEL,
    )
    def has_storage_permission(self, context: AuthedServiceContext, uid: UID) -> bool:
        return self.stash.has_storage_permission(
            StoragePermission(uid=uid, server_uid=context.server.id)
        )

    def has_read_permission(self, context: AuthedServiceContext, uid: UID) -> bool:
        return self.stash.has_permissions(
            [ActionObjectREAD(uid=uid, credentials=context.credentials)]
        )

    # not a public service endpoint
    @as_result(SyftException)
    def _user_code_execute(
        self,
        context: AuthedServiceContext,
        code_item: UserCode,
        kwargs: dict[str, Any],
        result_id: UID | None = None,
    ) -> ActionObjectPointer:
        override_execution_permission = (
            context.has_execute_permissions or context.role == ServiceRole.ADMIN
        )
        input_policy = code_item.get_input_policy(context)
        output_policy = code_item.get_output_policy(context)

        # Unwrap nested ActionObjects
        for _, arg in kwargs.items():
            self.flatten_action_arg(context, arg) if isinstance(arg, UID) else None

        if not override_execution_permission:
            if input_policy is None:
                if not code_item.is_output_policy_approved(context).unwrap():
                    raise SyftException(
                        public_message="Execution denied: Your code is waiting for approval"
                    )
                raise SyftException(
                    public_message=f"No input policy defined for user code: {code_item.id}"
                )

            # validate input policy, raises if not valid
            input_policy.is_valid(
                context=context,
                usr_input_kwargs=kwargs,
            )

            # Filter input kwargs based on policy
            filtered_kwargs = input_policy.filter_kwargs(
                kwargs=kwargs,
                context=context,
            )
        else:
            filtered_kwargs = retrieve_from_db(kwargs, context).unwrap()

        if hasattr(input_policy, "transform_kwargs"):
            filtered_kwargs = input_policy.transform_kwargs(  # type: ignore
                context,
                filtered_kwargs,
            ).unwrap()

        # update input policy to track any input state
        has_twin_inputs = False

        real_kwargs = {}
        for key, kwarg_value in filtered_kwargs.items():
            if isinstance(kwarg_value, TwinObject):
                has_twin_inputs = True
            real_kwargs[key] = kwarg_value

        result_id = UID() if result_id is None else result_id

        try:
            if not has_twin_inputs:
                # no twins
                # allow python types from inputpolicy
                filtered_kwargs = filter_twin_kwargs(
                    real_kwargs, twin_mode=TwinMode.NONE, allow_python_types=True
                ).unwrap()
                exec_result = execute_byte_code(code_item, filtered_kwargs, context)
                if exec_result.errored:
                    raise SyftException(public_message=exec_result.safe_error_message)

                if output_policy:
                    exec_result.result = output_policy.apply_to_output(
                        context,
                        exec_result.result,
                        update_policy=not override_execution_permission,
                    )
                code_item.output_policy = output_policy  # type: ignore
                context.server.services.user_code.update_code_state(context, code_item)
                if isinstance(exec_result.result, ActionObject):
                    result_action_object = ActionObject.link(
                        result_id=result_id, pointer_id=exec_result.result.id
                    )
                else:
                    result_action_object = wrap_result(result_id, exec_result.result)
            else:
                # twins
                private_kwargs = filter_twin_kwargs(
                    real_kwargs, twin_mode=TwinMode.PRIVATE, allow_python_types=True
                ).unwrap()
                private_exec_result = execute_byte_code(
                    code_item, private_kwargs, context
                )
                if private_exec_result.errored:
                    raise SyftException(
                        public_message=private_exec_result.safe_error_message
                    )

                if output_policy:
                    private_exec_result.result = output_policy.apply_to_output(
                        context,
                        private_exec_result.result,
                        update_policy=not override_execution_permission,
                    )
                code_item.output_policy = output_policy  # type: ignore
                context.server.services.user_code.update_code_state(context, code_item)
                result_action_object_private = wrap_result(
                    result_id, private_exec_result.result
                )

                mock_kwargs = filter_twin_kwargs(
                    real_kwargs, twin_mode=TwinMode.MOCK, allow_python_types=True
                ).unwrap()
                # relative
                from .action_data_empty import ActionDataEmpty

                if any(isinstance(v, ActionDataEmpty) for v in mock_kwargs.values()):
                    mock_exec_result_obj = ActionDataEmpty()
                else:
                    mock_exec_result = execute_byte_code(
                        code_item, mock_kwargs, context
                    )

                    if mock_exec_result.errored:
                        raise SyftException(
                            public_message=mock_exec_result.safe_error_message
                        )

                    if output_policy:
                        mock_exec_result.result = output_policy.apply_to_output(
                            context, mock_exec_result.result, update_policy=False
                        )
                    mock_exec_result_obj = mock_exec_result.result

                result_action_object_mock = wrap_result(result_id, mock_exec_result_obj)

                result_action_object = TwinObject(
                    id=result_id,
                    private_obj=result_action_object_private,
                    mock_obj=result_action_object_mock,
                )
        except Exception as e:
            raise SyftException.from_exception(e)

        return result_action_object

    # def raise_for_failed_execution(self, output: ExecutionOutput):
    #     if output.errored:
    #         raise SyftException(public_message="Execution of usercode failed, ask admin",
    #                                 private_message=output.stdout + "\n" + output.stderr)

    @as_result(SyftException)
    def set_result_to_store(
        self,
        result_action_object: ActionObject | TwinObject,
        context: AuthedServiceContext,
        output_policy: OutputPolicy | None = None,
        has_result_read_permission: bool = False,
    ) -> ActionObject:
        result_id = result_action_object.id
        # result_blob_id = result_action_object.syft_blob_storage_entry_id

        if output_policy is not None:
            output_readers = (
                output_policy.output_readers
                if not context.has_execute_permissions
                else []
            )
        else:
            output_readers = []

        # If flag is True, user has read permissions to the results in BlobStore
        if has_result_read_permission:
            output_readers.append(context.credentials)

        read_permission = ActionPermission.READ

        result_action_object._set_obj_location_(
            context.server.id,
            context.credentials,
        )
        blob_store_result: SyftResponseMessage = (
            result_action_object._save_to_blob_storage().unwrap()
        )
        if isinstance(blob_store_result, SyftWarning):
            logger.debug(blob_store_result.message)

        # IMPORTANT: DO THIS ONLY AFTER ._save_to_blob_storage
        if isinstance(result_action_object, TwinObject):
            result_blob_id = result_action_object.private.syft_blob_storage_entry_id
        else:
            result_blob_id = result_action_object.syft_blob_storage_entry_id  # type: ignore[unreachable]

        # pass permission information to the action store as extra kwargs
        # context.extra_kwargs = {"has_result_read_permission": True}

        # Since this just meta data about the result, they always have access to it.
        set_result = self._set(
            context,
            result_action_object,
            has_result_read_permission=True,
        ).unwrap()

        def store_permission(
            x: SyftVerifyKey | None = None,
        ) -> ActionObjectPermission:
            return ActionObjectPermission(result_id, read_permission, x)

        def blob_permission(
            x: SyftVerifyKey | None = None,
        ) -> ActionObjectPermission:
            return ActionObjectPermission(result_blob_id, read_permission, x)

        if len(output_readers) > 0:
            store_permissions = [store_permission(x) for x in output_readers]
            self.stash.add_permissions(store_permissions)

            if result_blob_id is not None:
                blob_permissions = [blob_permission(x) for x in output_readers]
                context.server.services.blob_storage.stash.add_permissions(
                    blob_permissions
                )

        return set_result

    @as_result(SyftException)
    def execute_plan(
        self,
        plan: Any,
        context: AuthedServiceContext,
        plan_kwargs: dict[str, ActionObject],
    ) -> ActionObject:
        id2inpkey = {v.id: k for k, v in plan.inputs.items()}

        for plan_action in plan.actions:
            if (
                hasattr(plan_action.remote_self, "id")
                and plan_action.remote_self.id in id2inpkey
            ):
                plan_action.remote_self = plan_kwargs[
                    id2inpkey[plan_action.remote_self.id]
                ]
            for i, arg in enumerate(plan_action.args):
                if arg in id2inpkey:
                    plan_action.args[i] = plan_kwargs[id2inpkey[arg]]

            for k, arg in enumerate(plan_action.kwargs):
                if arg in id2inpkey:
                    plan_action.kwargs[k] = plan_kwargs[id2inpkey[arg]]

        for plan_action in plan.actions:
            self.execute(context, plan_action)
        result_id = plan.outputs[0].id
        return self._get(
            context, result_id, TwinMode.NONE, has_permission=True
        ).unwrap()

    @as_result(SyftException)
    def call_function(
        self, context: AuthedServiceContext, action: Action
    ) -> ActionObject:
        # run function/class init
        _user_lib_config_registry = UserLibConfigRegistry.from_user(context.credentials)
        absolute_path = f"{action.path}.{action.op}"
        if absolute_path in _user_lib_config_registry:
            # TODO: implement properly
            # Now we are assuming its a function/class
            return execute_callable(self, context, action).unwrap()
        else:
            raise SyftException(
                public_message=f"Failed executing {action}. You have no permission for {absolute_path}"
            )

    @as_result(SyftException)
    def set_attribute(
        self,
        context: AuthedServiceContext,
        action: Action,
        resolved_self: ActionObject | TwinObject,
    ) -> TwinObject:
        args, _ = resolve_action_args(action, context, self).unwrap(
            public_message=f"Failed executing action {action} (could not resolve args)"
        )
        if not isinstance(args[0], ActionObject):
            raise SyftException(
                public_message=(
                    f"Failed executing action {action} setattribute requires"
                    " a non-twin string as first argument"
                )
            )
        name = args[0].syft_action_data
        # dont do the whole filtering dance with the name
        args = [args[1]]

        if isinstance(resolved_self, TwinObject):
            # todo, create copy?
            private_args = filter_twin_args(args, twin_mode=TwinMode.PRIVATE).unwrap()
            private_val = private_args[0]
            setattr(resolved_self.private.syft_action_data, name, private_val)
            # todo: what do we use as data for the mock here?
            # depending on permisisons?
            public_args = filter_twin_args(args, twin_mode=TwinMode.MOCK).unwrap()
            public_val = public_args[0]
            setattr(resolved_self.mock.syft_action_data, name, public_val)
            return TwinObject(
                id=action.result_id,
                private_obj=ActionObject.from_obj(
                    resolved_self.private.syft_action_data
                ),
                private_obj_id=action.result_id,
                mock_obj=ActionObject.from_obj(resolved_self.mock.syft_action_data),
                mock_obj_id=action.result_id,
            )

        else:
            # TODO: Implement for twinobject args
            args = filter_twin_args(args, twin_mode=TwinMode.NONE).unwrap()  # type: ignore[unreachable]
            val = args[0]
            setattr(resolved_self.syft_action_data, name, val)
            return (ActionObject.from_obj(resolved_self.syft_action_data),)
            # todo: permissions
            # setattr(resolved_self.syft_action_data, name, val)
            # val = resolved_self.syft_action_data
            # result_action_object = Ok(wrap_result(action.result_id, val))

    @as_result(SyftException)
    def get_attribute(
        self, action: Action, resolved_self: ActionObject | TwinObject
    ) -> TwinObject | ActionObject:
        if isinstance(resolved_self, TwinObject):
            private_result = getattr(resolved_self.private.syft_action_data, action.op)
            mock_result = getattr(resolved_self.mock.syft_action_data, action.op)
            return TwinObject(
                id=action.result_id,
                private_obj=ActionObject.from_obj(private_result),
                private_obj_id=action.result_id,
                mock_obj=ActionObject.from_obj(mock_result),
                mock_obj_id=action.result_id,
            )
        else:
            val = getattr(resolved_self.syft_action_data, action.op)  # type: ignore[unreachable]
            return wrap_result(action.result_id, val)

    @as_result(SyftException)
    def call_method(
        self,
        context: AuthedServiceContext,
        action: Action,
        resolved_self: ActionObject | TwinObject,
    ) -> TwinObject | Any:
        if isinstance(resolved_self, TwinObject):
            # method
            private_result = execute_object(
                self,
                context,
                resolved_self.private,
                action,
                twin_mode=TwinMode.PRIVATE,
            ).unwrap(public_message=f"Failed executing action {action}")
            mock_result = execute_object(
                self, context, resolved_self.mock, action, twin_mode=TwinMode.MOCK
            ).unwrap(public_message=f"Failed executing action {action}")

            return TwinObject(
                id=action.result_id,
                private_obj=private_result,
                private_obj_id=action.result_id,
                mock_obj=mock_result,
                mock_obj_id=action.result_id,
            )
        else:
            return execute_object(self, context, resolved_self, action).unwrap()  # type:ignore[unreachable]

    as_result(SyftException)

    def unwrap_nested_actionobjects(
        self, context: AuthedServiceContext, data: Any
    ) -> Any:
        """recursively unwraps nested action objects"""

        if isinstance(data, list):
            return [self.unwrap_nested_actionobjects(context, obj) for obj in data]

        if isinstance(data, dict):
            return {
                key: self.unwrap_nested_actionobjects(context, obj)
                for key, obj in data.items()
            }

        if isinstance(data, ActionObject):
            res = self.get(context=context, uid=data.id)

            nested_res = res.syft_action_data

            if isinstance(nested_res, ActionObject):
                raise SyftException(
                    public_message="More than double nesting of ActionObjects is currently not supported"
                )

            return nested_res

        return data

    def contains_nested_actionobjects(self, data: Any) -> bool:
        """
        returns if this is a list/set/dict that contains ActionObjects
        """

        def unwrap_collection(col: set | dict | list) -> [Any]:  # type: ignore
            return_values = []
            if isinstance(col, dict):
                values = list(col.values()) + list(col.keys())
            else:
                values = list(col)
            for v in values:
                if isinstance(v, list | dict | set):
                    return_values += unwrap_collection(v)
                else:
                    return_values.append(v)
            return return_values

        if isinstance(data, list | dict | set):
            values = unwrap_collection(data)
            has_action_object = any(isinstance(x, ActionObject) for x in values)
            return has_action_object
        elif isinstance(data, ActionObject):
            return True
        return False

    def flatten_action_arg(self, context: AuthedServiceContext, arg: UID) -> None:
        """ "If the argument is a collection (of collections) of ActionObjects,
        We want to flatten the collection and upload a new ActionObject that contins
        its values. E.g. [[ActionObject1, ActionObject2],[ActionObject3, ActionObject4]]
        -> [[value1, value2],[value3, value4]]
        """
        root_context = context.as_root_context()

        action_object = self.get(context=root_context, uid=arg)
        data = action_object.syft_action_data

        if self.contains_nested_actionobjects(data):
            new_data = self.unwrap_nested_actionobjects(context, data)
            # Update existing action object with the new flattened data
            action_object.syft_action_data_cache = new_data

            # we should create this with the permissions as the old object
            # currently its using the client verify key on the object
            action_object._save_to_blob_storage().unwrap()
            # we should create this with the permissions of the old object
            self._set(
                context=root_context,
                action_object=action_object,
            ).unwrap()

        return None

    @service_method(path="action.execute", name="execute", roles=GUEST_ROLE_LEVEL)
    def execute(self, context: AuthedServiceContext, action: Action) -> ActionObject:
        """Execute an operation on objects in the action store"""
        # relative
        from .plan import Plan

        if action.action_type == ActionType.CREATEOBJECT:
            result_action_object = action.create_object
        elif action.action_type == ActionType.SYFTFUNCTION:
            kwarg_ids = {}
            for k, v in action.kwargs.items():
                # transform lineage ids into ids
                kwarg_ids[k] = v.id
            return context.server.services.user_code._call(  # type: ignore[union-attr]
                context, action.user_code_id, action.result_id, **kwarg_ids
            ).unwrap()
        elif action.action_type == ActionType.FUNCTION:
            result_action_object = self.call_function(context, action).unwrap()
        else:
            resolved_self = self._get(
                context=context,
                uid=action.remote_self,
                twin_mode=TwinMode.NONE,
                has_permission=True,
            ).unwrap(
                public_message=f"Failed executing action {action}, could not resolve self: {action.remote_self}"
            )
            if action.op == "__call__" and resolved_self.syft_action_data_type == Plan:
                result_action_object = self.execute_plan(
                    plan=resolved_self.syft_action_data,
                    context=context,
                    plan_kwargs=action.kwargs,
                ).unwrap()
            elif action.action_type == ActionType.SETATTRIBUTE:
                result_action_object = self.set_attribute(
                    context, action, resolved_self
                ).unwrap()
            elif action.action_type == ActionType.GETATTRIBUTE:
                result_action_object = self.get_attribute(
                    action, resolved_self
                ).unwrap()
            elif action.action_type == ActionType.METHOD:
                result_action_object = self.call_method(
                    context, action, resolved_self
                ).unwrap()
            else:
                raise SyftException(public_message="unknown action")

        # check if we have read permissions on the result
        has_result_read_permission = self.has_read_permission_for_action_result(
            context, action
        )
        result_action_object._set_obj_location_(  # type: ignore[union-attr]
            context.server.id,
            context.credentials,
        )
        blob_store_result = result_action_object._save_to_blob_storage().unwrap()  # type: ignore[union-attr]
        # pass permission information to the action store as extra kwargs
        context.extra_kwargs = {
            "has_result_read_permission": has_result_read_permission
        }
        if isinstance(blob_store_result, SyftWarning):
            logger.debug(blob_store_result.message)
        set_result = self._set(
            context,
            result_action_object,
        )
        set_result = set_result.unwrap(
            public_message=f"Failed executing action {action}"
        )

        return set_result

    def has_read_permission_for_action_result(
        self, context: AuthedServiceContext, action: Action
    ) -> bool:
        action_obj_ids = (
            action.args + list(action.kwargs.values()) + [action.remote_self]
        )
        permissions = [
            ActionObjectREAD(uid=_id, credentials=context.credentials)
            for _id in action_obj_ids
        ]
        return self.stash.has_permissions(permissions)

    @service_method(path="action.exists", name="exists", roles=GUEST_ROLE_LEVEL)
    def exists(self, context: AuthedServiceContext, obj_id: UID) -> bool:
        """Checks if the given object id exists in the Action Store"""
        return self.stash.exists(context.credentials, obj_id)

    @service_method(
        path="action.delete",
        name="delete",
        roles=ADMIN_ROLE_LEVEL,
        unwrap_on_success=False,
    )
    def delete(
        self, context: AuthedServiceContext, uid: UID, soft_delete: bool = False
    ) -> SyftSuccess:
        obj = self.stash.get(uid=uid, credentials=context.credentials).unwrap()

        return_msg = []

        # delete any associated blob storage entry object to the action object
        blob_del_res = self._delete_blob_storage_entry(
            context=context, obj=obj
        ).unwrap()
        return_msg.append(blob_del_res.message)

        # delete the action object from the action store
        store_del_res = self._delete_from_action_store(
            context=context, uid=obj.id, soft_delete=soft_delete
        ).unwrap()

        return_msg.append(store_del_res.message)
        return SyftSuccess(message="\n".join(return_msg))

    @as_result(SyftException)
    def _delete_blob_storage_entry(
        self,
        context: AuthedServiceContext,
        obj: TwinObject | ActionObject,
    ) -> SyftSuccess:
        deleted_blob_ids = []

        if isinstance(obj, ActionObject) and obj.syft_blob_storage_entry_id:
            context.server.services.blob_storage.delete(
                context=context, uid=obj.syft_blob_storage_entry_id
            )
            deleted_blob_ids.append(obj.syft_blob_storage_entry_id)

        if isinstance(obj, TwinObject):
            if obj.private.syft_blob_storage_entry_id:
                context.server.services.blob_storage.delete(
                    context=context, uid=obj.private.syft_blob_storage_entry_id
                )
                deleted_blob_ids.append(obj.private.syft_blob_storage_entry_id)

            if obj.mock.syft_blob_storage_entry_id:
                context.server.services.blob_storage.delete(
                    context=context, uid=obj.mock.syft_blob_storage_entry_id
                )
                deleted_blob_ids.append(obj.mock.syft_blob_storage_entry_id)

        message = f"Deleted blob storage entries: {', '.join(str(blob_id) for blob_id in deleted_blob_ids)}"
        return SyftSuccess(message=message)

    @as_result(SyftException)
    def _delete_from_action_store(
        self,
        context: AuthedServiceContext,
        uid: UID,
        soft_delete: bool = False,
    ) -> SyftSuccess:
        if soft_delete:
            obj = self.stash.get(uid=uid, credentials=context.credentials).unwrap()

            if isinstance(obj, TwinObject):
                self._soft_delete_action_obj(
                    context=context, action_obj=obj.private
                ).unwrap()
                self._soft_delete_action_obj(
                    context=context, action_obj=obj.mock
                ).unwrap()
            if isinstance(obj, ActionObject):
                self._soft_delete_action_obj(context=context, action_obj=obj).unwrap()
        else:
            self.stash.delete_by_uid(credentials=context.credentials, uid=uid).unwrap()

        return SyftSuccess(message=f"Action object with uid '{uid}' deleted.")

    @as_result(SyftException)
    def _soft_delete_action_obj(
        self, context: AuthedServiceContext, action_obj: ActionObject
    ) -> ActionObject:
        action_obj.syft_action_data_cache = None
        action_obj._save_to_blob_storage().unwrap()
        return self._set(
            context=context,
            action_object=action_obj,
        ).unwrap()


@as_result(SyftException)
def resolve_action_args(
    action: Action, context: AuthedServiceContext, service: ActionService
) -> tuple[list, bool]:
    has_twin_inputs = False
    args = []
    for arg_id in action.args:
        arg_value = service._get(
            context=context, uid=arg_id, twin_mode=TwinMode.NONE, has_permission=True
        ).unwrap()
        if isinstance(arg_value, TwinObject):
            has_twin_inputs = True
        args.append(arg_value)
    return args, has_twin_inputs


@as_result(SyftException)
def resolve_action_kwargs(
    action: Action, context: AuthedServiceContext, service: ActionService
) -> tuple[dict, bool]:
    has_twin_inputs = False
    kwargs = {}
    for key, arg_id in action.kwargs.items():
        kwarg_value = service._get(
            context=context, uid=arg_id, twin_mode=TwinMode.NONE, has_permission=True
        ).unwrap()
        if isinstance(kwarg_value, TwinObject):
            has_twin_inputs = True
        kwargs[key] = kwarg_value
    return kwargs, has_twin_inputs


@as_result(SyftException)
def execute_callable(
    service: ActionService,
    context: AuthedServiceContext,
    action: Action,
) -> ActionObject:
    args, has_arg_twins = resolve_action_args(action, context, service).unwrap()
    kwargs, has_kwargs_twins = resolve_action_kwargs(action, context, service).unwrap()
    has_twin_inputs = has_arg_twins or has_kwargs_twins
    # 🔵 TODO 10: Get proper code From old RunClassMethodAction to ensure the function
    # is not bound to the original object or mutated

    # TODO: get from CMPTree is probably safer
    def _get_target_callable(path: str, op: str) -> Any:
        path_elements = path.split(".")
        res = importlib.import_module(path_elements[0])
        for p in path_elements[1:]:
            res = getattr(res, p)
        res = getattr(res, op)
        return res

    target_callable = _get_target_callable(action.path, action.op)

    result = None
    if not target_callable:
        raise SyftException(public_message="No target callable found")

    if not has_twin_inputs:
        # if twin_mode == TwinMode.NONE and not has_twin_inputs:
        twin_mode = TwinMode.NONE
        # no twins
        filtered_args = filter_twin_args(args, twin_mode=twin_mode).unwrap()
        filtered_kwargs = filter_twin_kwargs(kwargs, twin_mode=twin_mode).unwrap()
        result = target_callable(*filtered_args, **filtered_kwargs)
        result_action_object = wrap_result(action.result_id, result)
    else:
        twin_mode = TwinMode.PRIVATE
        private_args = filter_twin_args(args, twin_mode=twin_mode).unwrap()
        private_kwargs = filter_twin_kwargs(kwargs, twin_mode=twin_mode).unwrap()
        private_result = target_callable(*private_args, **private_kwargs)
        result_action_object_private = wrap_result(action.result_id, private_result)

        twin_mode = TwinMode.MOCK
        mock_args = filter_twin_args(args, twin_mode=twin_mode).unwrap()
        mock_kwargs = filter_twin_kwargs(kwargs, twin_mode=twin_mode).unwrap()
        mock_result = target_callable(*mock_args, **mock_kwargs)
        result_action_object_mock = wrap_result(action.result_id, mock_result)

        result_action_object = TwinObject(
            id=action.result_id,
            private_obj=result_action_object_private,
            mock_obj=result_action_object_mock,
        )

    return result_action_object


@as_result(SyftException)
def execute_object(
    service: ActionService,
    context: AuthedServiceContext,
    resolved_self: ActionObject,
    action: Action,
    twin_mode: TwinMode = TwinMode.NONE,
) -> TwinObject | ActionObject:
    unboxed_resolved_self = resolved_self.syft_action_data
    args, has_arg_twins = resolve_action_args(action, context, service).unwrap()

    kwargs, has_kwargs_twins = resolve_action_kwargs(action, context, service).unwrap()
    has_twin_inputs = has_arg_twins or has_kwargs_twins

    # 🔵 TODO 10: Get proper code From old RunClassMethodAction to ensure the function
    # is not bound to the original object or mutated
    target_method = getattr(unboxed_resolved_self, action.op, None)
    result = None

    if not target_method:
        raise SyftException(public_message="could not find target method")
    if twin_mode == TwinMode.NONE and not has_twin_inputs:
        # no twins
        filtered_args = filter_twin_args(args, twin_mode=twin_mode).unwrap()
        filtered_kwargs = filter_twin_kwargs(kwargs, twin_mode=twin_mode).unwrap()
        result = target_method(*filtered_args, **filtered_kwargs)
        result_action_object = wrap_result(action.result_id, result)
    elif twin_mode == TwinMode.NONE and has_twin_inputs:
        # self isn't a twin but one of the inputs is
        private_args = filter_twin_args(args, twin_mode=TwinMode.PRIVATE).unwrap()
        private_kwargs = filter_twin_kwargs(kwargs, twin_mode=TwinMode.PRIVATE).unwrap()
        private_result = target_method(*private_args, **private_kwargs)
        result_action_object_private = wrap_result(action.result_id, private_result)

        mock_args = filter_twin_args(args, twin_mode=TwinMode.MOCK).unwrap()
        mock_kwargs = filter_twin_kwargs(kwargs, twin_mode=TwinMode.MOCK).unwrap()
        mock_result = target_method(*mock_args, **mock_kwargs)
        result_action_object_mock = wrap_result(action.result_id, mock_result)

        result_action_object = TwinObject(
            id=action.result_id,
            private_obj=result_action_object_private,
            mock_obj=result_action_object_mock,
        )
    elif twin_mode == twin_mode.PRIVATE:  # type:ignore
        # twin private path
        private_args = filter_twin_args(args, twin_mode=twin_mode).unwrap()  # type:ignore[unreachable]
        private_kwargs = filter_twin_kwargs(kwargs, twin_mode=twin_mode).unwrap()
        result = target_method(*private_args, **private_kwargs)
        result_action_object = wrap_result(action.result_id, result)
    elif twin_mode == twin_mode.MOCK:  # type:ignore
        # twin mock path
        mock_args = filter_twin_args(args, twin_mode=twin_mode).unwrap()  # type:ignore[unreachable]
        mock_kwargs = filter_twin_kwargs(kwargs, twin_mode=twin_mode).unwrap()
        target_method = getattr(unboxed_resolved_self, action.op, None)
        result = target_method(*mock_args, **mock_kwargs)
        result_action_object = wrap_result(action.result_id, result)
    else:
        raise SyftException(
            public_message=f"Bad combination of: twin_mode: {twin_mode} and has_twin_inputs: {has_twin_inputs}"
        )

    return result_action_object


def wrap_result(result_id: UID, result: Any) -> ActionObject:
    # 🟡 TODO 11: Figure out how we want to store action object results
    action_type = action_type_for_type(result)
    result_action_object = action_type(id=result_id, syft_action_data_cache=result)
    return result_action_object


@as_result(SyftException)
def filter_twin_args(args: list[Any], twin_mode: TwinMode) -> Any:
    filtered = []
    for arg in args:
        if isinstance(arg, TwinObject):
            if twin_mode == TwinMode.PRIVATE:
                filtered.append(arg.private.syft_action_data)
            elif twin_mode == TwinMode.MOCK:
                filtered.append(arg.mock.syft_action_data)
            else:
                raise SyftException(
                    public_message=f"Filter can only use {TwinMode.PRIVATE} or {TwinMode.MOCK}"
                )
        else:
            filtered.append(arg.syft_action_data)
    return filtered


@as_result(SyftException)
def filter_twin_kwargs(
    kwargs: dict, twin_mode: TwinMode, allow_python_types: bool = False
) -> Any:
    filtered = {}
    for k, v in kwargs.items():
        if isinstance(v, TwinObject):
            if twin_mode == TwinMode.PRIVATE:
                filtered[k] = v.private.syft_action_data
            elif twin_mode == TwinMode.MOCK:
                filtered[k] = v.mock.syft_action_data
            else:
                raise SyftException(
                    public_message=f"Filter can only use {TwinMode.PRIVATE} or {TwinMode.MOCK}"
                )
        else:
            if isinstance(v, ActionObject):
                filtered[k] = v.syft_action_data
            elif (
                isinstance(v, str | int | float | dict | CustomEndpointActionObject)
                and allow_python_types
            ):
                filtered[k] = v
            else:
                raise SyftException(
                    public_message=f"unexepected value {v} passed to filtered twin kwargs"
                )
    return filtered


TYPE_TO_SERVICE[ActionObject] = ActionService
TYPE_TO_SERVICE[TwinObject] = ActionService
TYPE_TO_SERVICE[AnyActionObject] = ActionService

SERVICE_TO_TYPES[ActionService].update({ActionObject, TwinObject, AnyActionObject})
