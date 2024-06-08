# stdlib
from typing import Any

# third party
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator
from typing_extensions import Self

# relative
from ...client.api import NodeIdentity
from ...client.client import SyftClient
from ...client.client import SyftClientSessionCache
from ...types.uid import UID
from ...util import options
from ...util.colors import SURFACE
from ..code.user_code import SubmitUserCode
from ..code.user_code import UserCode
from ..enclave.enclave import EnclaveInstance
from ..metadata.node_metadata import NodeMetadata
from ..request.request import Request
from ..request.request import RequestStatus
from ..response import SyftError
from ..response import SyftException
from .project import Project
from .project import ProjectRequest
from .project import ProjectSubmit


class DistributedProject(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str = ""
    code: UserCode | SubmitUserCode  # only one code per project for this prototype
    clients: dict[UID, SyftClient] = Field(default_factory=dict)
    members: dict[UID, NodeIdentity] = Field(default_factory=dict)
    all_projects: list[Project] = Field(default_factory=list)
    project_permissions: set[str] = Field(default_factory=set)  # Unused at the moment

    def _coll_repr_(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "pending requests": self.pending_requests,
        }

    def _repr_html_(self) -> str:
        return (
            f"""
            <style>
            .syft-project {{color: {SURFACE[options.color_theme]};}}
            </style>
            """
            + "<div class='syft-project'>"
            + f"<h3>{self.name}</h3>"
            + f"<p>{self.description}</p>"
            + self.requests._repr_html_()
            + "</div>"
        )

    @classmethod
    def get_by_name(cls, name: str, clients: list[SyftClient]) -> Self:
        all_projects = []
        for client in clients:
            project = client.projects.get_by_name(name)
            if isinstance(project, SyftError):
                raise SyftException(project.message)
            all_projects.append(project)

        # TODO verify that all the projects in the `all_projects` list are the same
        project = all_projects[0]
        description = project.description
        code = project.requests[0].code  # TODO fix possible errors
        return cls(
            name=name, description=description, code=code, all_projects=all_projects
        )

    @property
    def requests(self) -> list[Request]:
        requests: list[Request] = []
        for project in self.all_projects:
            requests.extend(
                event.request
                for event in project.events
                if isinstance(event, ProjectRequest)
            )
        return requests

    @property
    def pending_requests(self) -> int:
        return sum(
            [request.status == RequestStatus.PENDING for request in self.requests]
        )

    @field_validator("code", mode="before")
    @classmethod
    def verify_code(cls, code: SubmitUserCode) -> SubmitUserCode:
        if not code.deployment_policy_init_kwargs:
            raise ValueError("Deployment policy not found in code.")
        provider = code.deployment_policy_init_kwargs.get("provider")
        if not provider:
            raise ValueError("Provider not found in deployment policy.")
        if not isinstance(provider, EnclaveInstance):
            raise SyftException(
                "Only `EnclaveInstance` is supported as provider for now."
            )
        return code

    @field_validator("clients", mode="before")
    @classmethod
    def verify_clients(cls, val: list[SyftClient]) -> list[SyftClient]:
        # SyftClients must be logged in by the same emails
        if len(val) > 0:
            emails = {client.logged_in_user for client in val}
            if len(emails) > 1:
                raise ValueError(
                    f"All clients must be logged in from the same account. Found multiple: {emails}"
                )
        return val

    @model_validator(mode="after")
    def _populate_auto_generated_fields(self) -> Self:
        self.clients = self._get_clients_from_code()
        self.members = self._get_members_from_clients()
        return self

    def submit(self) -> Self:
        self._pre_submit_checks()
        projects_map = self._submit_project_to_all_clients()
        self.all_projects = list(projects_map.values())
        return self

    def request_execution(self, blocking: bool = True) -> Any:
        # TODO
        # 1. Check that pending_requests is 0, meaning all requests have been accepted
        # 2. Request Enclave owner for access
        # 3. Wait for Enclave to be set up by the owner domain (project, code and users)
        # 4. Request each domain to transfer their assets to the Enclave
        # 5. Execute the code on the Enclave
        # 6. Return the results
        # 7. Cleanup the Enclave
        pass

    def _get_clients_from_code(self) -> dict[UID, SyftClient]:
        if not self.code or not self.code.input_policy_init_kwargs:
            return {}

        clients = {
            policy.node_id: client
            for policy in self.code.input_policy_init_kwargs.keys()
            if (
                client := SyftClientSessionCache.get_client_for_node_uid(policy.node_id)
            )
        }
        return clients

    def _get_members_from_clients(self) -> dict[UID, NodeIdentity]:
        return {
            node_id: self._to_node_identity(client)
            for node_id, client in self.clients.items()
        }

    @staticmethod
    def _to_node_identity(client: SyftClient) -> NodeIdentity:
        if isinstance(client, SyftClient) and client.metadata is not None:
            metadata = client.metadata.to(NodeMetadata)
            return metadata.to(NodeIdentity)
        else:
            raise SyftException(f"members must be SyftClient. Received: {type(client)}")

    def _pre_submit_checks(self) -> bool:
        try:
            # Check if the user can create projects
            for client in self.clients.values():
                result = client.api.services.project.can_create_project()
                if isinstance(result, SyftError):
                    raise SyftException(result.message)
        except Exception:
            raise SyftException("Only Data Scientists can create projects")

        return True

    def _submit_project_to_all_clients(self) -> dict[SyftClient, Project]:
        projects_map: dict[SyftClient, Project] = {}
        for client in self.clients.values():
            # Creating projects and code requests separately across each client
            # TODO handle failures in between
            new_project = ProjectSubmit(
                name=self.name,
                description=self.description,
                members=[client],
            )
            new_project.create_code_request(self.code, client)
            project = new_project.send()
            projects_map[client] = project[0] if isinstance(project, list) else project
        return projects_map
