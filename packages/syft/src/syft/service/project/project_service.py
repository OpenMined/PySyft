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
        result = self.stash.get_all(context.credentials)
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
            project.id = project_id
            project_obj = project.to(NewProject, context=context)
            # project_obj.id = project_id

            result = self.stash.set(context.credentials, project_obj)
            if result.is_err():
                return SyftError(message=str(result.err()))
            return result.ok()
        except Exception as e:
            print("Failed to submit Project", e)
            raise e

    @service_method(
        path="newproject.add_event",
        name="add_event",
        roles=GUEST_ROLE_LEVEL,
    )
    def add_event(
        self, context: AuthedServiceContext, project_event: ProjectEvent
    ) -> Union[SyftSuccess, SyftError]:
        """To add events to a projects"""
        # Event object should be received from the leader of the project

        # retrieve the project object by node verify key
        project_obj = self.stash.get_by_uid(
            context.node.verify_key, uid=project_event.project_id
        )

        if project_obj.is_ok():
            project: NewProject = project_obj.ok()
            if project.state_sync_leader.verify_key == context.node.verify_key:
                return SyftError(
                    message="Project Events should be passed to leader by broadcast endpoint"
                )
            if context.credentials != project.state_sync_leader.verify_key:
                return SyftError(
                    message="Only the leader of the project can add events"
                )

            project.events.append(project_event)
            # updating the project object using root verify key of node
            result = self.stash.update(context.node.verify_key, project)

            if result.is_err():
                return SyftError(message=str(result.err()))
            return SyftSuccess(
                message=f"Project event {project_event.id} added successfully "
            )

        if project_obj.is_err():
            return SyftError(message=str(project_obj.err()))

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

            # Broadcast the event to all the members of the project
            network_service = context.node.get_service("networkservice")
            for sharedholder in project.shareholders:
                if sharedholder.verify_key != context.node.verify_key:
                    # Retrieving the NodePeer Object to communicate with the node
                    peer = network_service.stash.get_for_verify_key(
                        credentials=context.node.verify_key,
                        verify_key=sharedholder.verify_key,
                    )

                    if peer.is_err():
                        return SyftError(
                            message=f"Leader node does not have peer {sharedholder.name}-{sharedholder.id}"
                            + " Kindly exchange routes with the peer"
                        )
                    peer = peer.ok()
                    client = peer.client_with_context(context)
                    event_result = client.api.services.newproject.add_event(
                        project_event
                    )
                    if isinstance(event_result, SyftError):
                        return event_result

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
        result = self.stash.get_all(context.credentials)
        if result.is_err():
            return SyftError(message=str(result.err()))
        projects = result.ok()
        return projects


TYPE_TO_SERVICE[Project] = ProjectService
TYPE_TO_SERVICE[NewProject] = NewProjectService
SERVICE_TO_TYPES[ProjectService].update({Project})
SERVICE_TO_TYPES[NewProjectService].update({NewProject})
