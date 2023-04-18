# stdlib
from enum import Enum
from typing import Callable
from typing import List
from typing import Optional
from typing import Type

# third party
from result import OkErr

# relative
from ...node.credentials import SyftVerifyKey
from ...serde.serializable import serializable
from ...store.linked_obj import LinkedObject
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SyftObject
from ...types.transforms import TransformContext
from ...types.transforms import generate_id
from ...types.transforms import transform
from ...types.uid import UID
from ..code.user_code import UserCode
from ..request.request import EnumMutation
from ..request.request import Request
from ..request.request import SubmitRequest
from ..request.request import UserCodeStatusChange
from ..request.request_service import RequestService
from ..response import SyftError
from ..service import TYPE_TO_SERVICE


@serializable()
class Project(SyftObject):
    __canonical_name__ = "Project"
    __version__ = SYFT_OBJECT_VERSION_1

    name: str
    description: str
    user_verify_key: SyftVerifyKey
    requests: Optional[List[Request]]

    __attr_searchable__ = [
        "user_verify_key",
        "name",
    ]
    __attr_unique__ = ["name"]

    __attr_repr_cols__ = ["requests"]


@serializable()
class ObjectPermissionChange(SyftObject):
    __canonical_name__ = "PermissionChange"
    __version__ = SYFT_OBJECT_VERSION_1

    object_uid: UID
    permission: Enum
    object_type: Type[SyftObject]


@serializable()
class ProjectSubmit(SyftObject):
    __canonical_name__ = "ProjectSubmit"
    __version__ = SYFT_OBJECT_VERSION_1

    id: Optional[UID]
    name: str
    description: Optional[str]
    changes: List[ObjectPermissionChange] = []

    __attr_repr_cols__ = ["name", "changes"]

    def set_description(self, description: str, msg: Optional[str] = None) -> None:
        self.description = description

    def add_request(self, obj: SyftObject, permission: Enum) -> None:
        change = ObjectPermissionChange(
            object_uid=obj.id, object_type=type(obj), permission=permission
        )
        self.changes.append(change)


def submit_changes(context: TransformContext) -> TransformContext:
    changes = context.output.pop("changes", [])
    if changes:
        mutations = []
        for change in changes:
            service_type = TYPE_TO_SERVICE[change.object_type]
            linked_obj = LinkedObject(
                node_uid=context.node.id,
                service_type=service_type,
                object_type=change.object_type,
                object_uid=change.object_uid,
            )

            if change.object_type == UserCode:
                mutation = UserCodeStatusChange(
                    value=change.permission, linked_obj=linked_obj
                )
            else:
                mutation = EnumMutation.from_obj(
                    linked_obj=linked_obj, attr_name="status", value=change.permission
                )

            mutations.append(mutation)
        submit_request = SubmitRequest(changes=mutations)
        request_submit_method = context.node.get_service_method(RequestService.submit)
        result = request_submit_method(
            context=context, request=submit_request, send_message=False
        )

        if isinstance(result, SyftError):
            return result
        if isinstance(result, OkErr):
            result = result.ok()
        context.output["requests"] = [result]
    context.output["user_verify_key"] = context.credentials
    return context


@transform(ProjectSubmit, Project)
def projectsubmit_to_project() -> List[Callable]:
    return [generate_id, submit_changes]
