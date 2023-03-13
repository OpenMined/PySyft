# stdlib
from typing import List
from typing import Union

# third party
from result import Err
from result import Ok

# relative
from ....telemetry import instrument
from .context import AuthedServiceContext
from .document_store import DocumentStore
from .linked_obj import LinkedObject
from .message_service import CreateMessage
from .message_service import Message
from .message_service import MessageService
from .project import Project
from .project import ProjectSubmit
from .project_stash import ProjectStash
from .response import SyftError
from .response import SyftSuccess
from .serializable import serializable
from .service import AbstractService
from .service import SERVICE_TO_TYPES
from .service import TYPE_TO_SERVICE
from .service import service_method
from .user_service import UserService


@instrument
@serializable(recursive_serde=True)
class ProjectService(AbstractService):
    store: DocumentStore
    stash: ProjectStash

    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.stash = ProjectStash(store=store)

    @service_method(path="project.submit", name="submit")
    def submit(
        self, context: AuthedServiceContext, project: ProjectSubmit
    ) -> Union[SyftSuccess, SyftError]:
        """Submit a Project"""
        try:
            result = self.stash.set(project.to(Project, context=context))

            if result.is_ok():
                result = result.ok()
                link = LinkedObject.with_context(result, context=context)
                admin_verify_key = context.node.get_service_method(
                    UserService.admin_verify_key
                )

                root_verify_key = admin_verify_key()

                message = CreateMessage(
                    subject="Project Approval",
                    from_user_verify_key=context.credentials,
                    to_user_verify_key=root_verify_key,
                    linked_obj=link,
                )
                method = context.node.get_service_method(MessageService.send)
                result = method(context=context, message=message)
                if isinstance(result, Message):
                    result = Ok(SyftSuccess(message="Project Submitted"))
                else:
                    result = Err(result)

            if result.is_err():
                return SyftError(message=str(result.err()))
            return result.ok()
        except Exception as e:
            print("Failed to submit Project", e)
            raise e

    @service_method(path="project.get_all", name="get_all")
    def get_all(self, context: AuthedServiceContext) -> Union[List[Project], SyftError]:
        result = self.stash.get_all_for_verify_key(verify_key=context.credentials)
        if result.is_err():
            return SyftError(message=str(result.err()))
        projects = result.ok()
        return projects


TYPE_TO_SERVICE[Project] = ProjectService
SERVICE_TO_TYPES[ProjectService].update({Project})
