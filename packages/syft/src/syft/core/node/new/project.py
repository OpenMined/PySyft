# stdlib
from typing import Callable
from typing import List
from typing import Optional

# relative
from ....core.node.common.node_table.syft_object import SYFT_OBJECT_VERSION_1
from ....core.node.common.node_table.syft_object import SyftObject
from ...common.serde.serializable import serializable
from ...common.uid import UID
from .action_store import ActionPermission
from .credentials import SyftVerifyKey
from .request import ActionStoreChange
from .request import Request
from .request import SubmitRequest
from .request_service import RequestService
from .response import SyftError
from .transforms import TransformContext
from .transforms import generate_id
from .transforms import transform


@serializable(recursive_serde=True)
class Project(SyftObject):
    __canonical_name__ = "Project"
    __version__ = SYFT_OBJECT_VERSION_1

    name: str
    description: str
    user_verify_key: SyftVerifyKey
    request: Optional[Request]

    __attr_searchable__ = [
        "user_verify_key",
        "name",
    ]
    __attr_unique__ = ["name"]


@serializable(recursive_serde=True)
class ProjectSubmit(SyftObject):
    __canonical_name__ = "ProjectSubmit"
    __version__ = SYFT_OBJECT_VERSION_1

    id: Optional[UID]
    name: str
    description: Optional[str]
    changes: Optional[List[ActionStoreChange]]

    def set_description(self, description: str, msg: Optional[str] = None) -> None:
        self.description = description

    def add_request(
        self, permission: ActionPermission, action_object: SyftObject
    ) -> None:
        change = ActionStoreChange(
            action_object_uid=action_object.id, apply_permission_type=permission
        )
        self.changes.append(change)


def save_changes(context: TransformContext) -> TransformContext:
    changes = context.output.pop("changes", [])
    if changes:
        submit_request = SubmitRequest(changes=changes)
        request_submit_method = context.node.get_service_method(RequestService.submit)
        result = request_submit_method(request=submit_request)
        if isinstance(result, SyftError):
            return result
        context.output["request"] = result
    context.output["user_verify_key"] = context.credentials
    return context


@transform(ProjectSubmit, Project)
def projectsubmit_to_project() -> List[Callable]:
    return [generate_id, save_changes]
