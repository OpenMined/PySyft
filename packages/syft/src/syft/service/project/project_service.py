# stdlib
from typing import List
from typing import Union

# third party
from result import Err
from result import Ok

# relative
from ...serde.serializable import serializable
from ...store.document_store import DocumentStore
from ...store.linked_obj import LinkedObject
from ...types.uid import UID
from ...util.telemetry import instrument
from ..context import AuthedServiceContext
from ..message.message_service import CreateMessage
from ..message.message_service import Message
from ..message.message_service import MessageService
from ..response import SyftError
from ..response import SyftSuccess
from ..service import AbstractService
from ..service import SERVICE_TO_TYPES
from ..service import TYPE_TO_SERVICE
from ..service import service_method
from ..user.user_roles import GUEST_ROLE_LEVEL
from ..user.user_service import UserService
from .project import NewProject
from .project import NewProjectSubmit
from .project import Project
from .project import ProjectEvent
from .project import ProjectSubmit
from .project_stash import NewProjectStash
from .project_stash import ProjectStash


@instrument
@serializable()
class ProjectService(AbstractService):
    store: DocumentStore
    stash: ProjectStash

    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.stash = ProjectStash(store=store)

    @service_method(path="project.submit", name="submit", roles=GUEST_ROLE_LEVEL)
    def submit(
        self, context: AuthedServiceContext, project: ProjectSubmit
    ) -> Union[SyftSuccess, SyftError]:
        """Submit a Project"""
        try:
            result = self.stash.set(
                context.credentials, project.to(Project, context=context)
            )

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


@instrument
@serializable()
class NewProjectService(AbstractService):
    store: DocumentStore
    stash: NewProjectStash

    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.stash = NewProjectStash(store=store)

    @service_method(
        path="newproject.create_project", name="create_project", roles=GUEST_ROLE_LEVEL
    )
    def create_project(
        self,
        context: AuthedServiceContext,
        project: NewProjectSubmit,
        project_id: UID,
    ) -> Union[SyftSuccess, SyftError]:
        """Start a Project"""
        try:
            project_obj = project.to(NewProject, context=context)
            project_obj.id = project_id

            result = self.stash.set(context.credentials, project_obj)
            if result.is_err():
                return SyftError(message=str(result.err()))
            return result.ok()
        except Exception as e:
            print("Failed to submit Project", e)
            raise e

    @service_method(
        path="newproject.broadcast_event",
        name="broadcast_event",
        roles=GUEST_ROLE_LEVEL,
    )
    def broadcast_event(
        self, context: AuthedServiceContext, project_event: ProjectEvent
    ) -> Union[SyftSuccess, SyftError]:
        """To add events to a projects"""
        # Only the leader of the project could add events to the projects
        # Any Event to be added to the project should be sent to the leader of the project
        # The leader broadcasts the event to all the members of the project
        project_obj = self.stash.get_by_uid(
            context.credentials, uid=project_event.project_id
        )

        if project_obj.is_ok():
            project: NewProject = project_obj.ok()
            if project.state_sync_leader.verify_key != context.node.verify_key:
                return SyftError(
                    message="Only the leader of the project can broadcast events"
                )

            project.events.append(project_event)
            result = self.stash.update(context.credentials, project)

            if result.is_err():
                return SyftError(message=str(result.err()))
            return result.ok()

        if project_obj.is_err():
            return SyftError(message=str(project_obj.err()))

    @service_method(path="newproject.get_all", name="get_all")
    def get_all(
        self, context: AuthedServiceContext
    ) -> Union[List[NewProject], SyftError]:
        result = self.stash.get_all_for_verify_key(verify_key=context.credentials)
        if result.is_err():
            return SyftError(message=str(result.err()))
        projects = result.ok()
        return projects


TYPE_TO_SERVICE[Project] = ProjectService
TYPE_TO_SERVICE[NewProject] = NewProjectService
SERVICE_TO_TYPES[ProjectService].update({Project})
SERVICE_TO_TYPES[NewProjectService].update({NewProject})
