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
from ...client.api import ServerIdentity
from ...client.client import SyftClient
from ...client.client import SyftClientSessionCache
from ...types.uid import UID
from ...util import options
from ...util.colors import SURFACE
from ...util.util import human_friendly_join
from ..code.user_code import SubmitUserCode
from ..code.user_code import UserCode
from ..enclave.enclave import EnclaveInstance
from ..metadata.server_metadata import ServerMetadata
from ..request.request import Request
from ..request.request import RequestStatus
from ..response import SyftError
from ..response import SyftException
from .project import Project
from .project import ProjectRequest
from .project import ProjectSubmit


class DistributedProject(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    id: UID = Field(default_factory=UID)
    name: str
    description: str = ""
    code: UserCode | SubmitUserCode  # only one code per project for this prototype
    clients: dict[UID, SyftClient] = Field(default_factory=dict)
    members: dict[UID, ServerIdentity] = Field(default_factory=dict)
    all_projects: dict[SyftClient, Project] = Field(default_factory=dict)
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
        all_projects = {}
        for client in clients:
            project = client.projects.get_by_name(name)
            if isinstance(project, SyftError):
                raise SyftException(project.message)
            all_projects[client] = project

        # TODO verify that DS passed all clients in the args correctly, else raise exception
        # TODO verify that all the projects in the `all_projects` list are the same
        project = next(iter(all_projects.values()))
        description = project.description
        code = project.requests[0].code  # TODO fix possible errors
        return cls(
            name=name, description=description, code=code, all_projects=all_projects
        )

    @property
    def requests(self) -> list[Request]:
        requests: list[Request] = []
        for project in self.all_projects.values():
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
    def verify_code(cls, code: UserCode | SubmitUserCode) -> UserCode | SubmitUserCode:
        if not code.runtime_policy_init_kwargs:
            raise ValueError("Runtime policy not found in code.")
        provider = code.runtime_policy_init_kwargs.get("provider")
        if not provider:
            raise ValueError("Provider not found in runtime policy.")
        if not isinstance(provider, EnclaveInstance):
            raise SyftException(
                "Only `EnclaveInstance` is supported as provider for now."
            )
        if isinstance(code, SubmitUserCode) and not code.id:
            code.id = UID()
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
        self.all_projects = self._submit_project_to_all_clients()
        return self

    def request_execution(self, blocking: bool = True) -> Any:
        self._pre_execution_request_checks()
        code = self.verify_code(self.code)

        # Request Enclave to be set up by its owner datasite
        provider = code.runtime_policy_init_kwargs.get("provider")
        owner_server_id = provider.syft_server_location
        owner_client = self.clients.get(owner_server_id)
        if not owner_client:
            raise SyftException(
                f"Can't access Syft client. You must login to {self.syft_server_location}"
            )
        enclave_code_created = owner_client.api.services.enclave.request_enclave(
            user_code_id=self.code.id
        )
        if isinstance(enclave_code_created, SyftError):
            raise SyftException(enclave_code_created.message)

        # Request each datasite to transfer their assets to the Enclave
        for client in self.clients.values():
            assets_transferred = client.api.services.enclave.request_assets_upload(
                user_code_id=self.code.id
            )
            if isinstance(assets_transferred, SyftError):
                raise SyftException(assets_transferred.message)
            print(assets_transferred.message)

        result_parts = []
        for client in self.clients.values():
            result = client.api.services.enclave.request_code_execution(
                user_code_id=self.code.id
            )
            if isinstance(result, SyftError):
                return SyftError(message=f"Enclave execution failure: {result.message}")
            else:
                result_parts.append(result)

        return result_parts[0]

    def _get_clients_from_code(self) -> dict[UID, SyftClient]:
        if not self.code or not self.code.input_policy_init_kwargs:
            return {}

        clients = {
            policy.server_id: client
            for policy in self.code.input_policy_init_kwargs.keys()
            if
            (
                # TODO use server_uid, verify_key instead as there could be multiple logged-in users to the same client
                client := SyftClientSessionCache.get_client_for_server_uid(
                    policy.server_id
                )
            )
        }
        return clients

    def _get_members_from_clients(self) -> dict[UID, ServerIdentity]:
        return {
            server_id: self._to_server_identity(client)
            for server_id, client in self.clients.items()
        }

    @staticmethod
    def _to_server_identity(client: SyftClient) -> ServerIdentity:
        if isinstance(client, SyftClient) and client.metadata is not None:
            metadata = client.metadata.to(ServerMetadata)
            return metadata.to(ServerIdentity)
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
                id=self.id,
                name=self.name,
                description=self.description,
                members=[client],
            )
            new_project.create_code_request(self.code, client)
            project = new_project.send()
            projects_map[client] = project[0] if isinstance(project, list) else project
        return projects_map

    def _pre_execution_request_checks(self) -> bool:
        members_servers_pending_approval = [
            request.syft_server_location
            for request in self.requests
            if request.status == RequestStatus.PENDING
        ]
        if members_servers_pending_approval:
            member_names = [
                self._get_server_name(member_server_id)
                or f"Server ID: {member_server_id}"
                for member_server_id in members_servers_pending_approval
            ]
            raise SyftException(
                f"Cannot execute project as approval request is pending for {human_friendly_join(member_names)}."
            )
        return True

    def _get_server_name(self, server_id: UID) -> str | None:
        server_identity = self.members.get(server_id)
        return server_identity.server_name if server_identity else None
