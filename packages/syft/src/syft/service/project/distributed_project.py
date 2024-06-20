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
from ...util.util import human_friendly_join
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

    @model_validator(mode="after")
    def _set_code_uid(self) -> Self:
        self.code.id = UID()
        return self

    def submit(self) -> Self:
        self._pre_submit_checks()
        self.all_projects = self._submit_project_to_all_clients()
        return self

    def request_execution(self, blocking: bool = True) -> Any:
        self._pre_execution_request_checks()
        code = self.verify_code(self.code)

        # Request Enclave to be set up by its owner domain
        provider = code.deployment_policy_init_kwargs.get("provider")
        owner_node_id = provider.syft_node_location
        owner_client = self.clients.get(owner_node_id)
        if not owner_client:
            raise SyftException(
                f"Can't access Syft client. You must login to {self.syft_node_location}"
            )
        enclave_code_created = (
            owner_client.api.services.enclave.request_enclave_for_code_execution(
                service_func_name=self.code.service_func_name
            )
        )
        if isinstance(enclave_code_created, SyftError):
            raise SyftException(enclave_code_created.message)

        # Request each domain to transfer their assets to the Enclave
        for client in self.clients.values():
            assets_transferred = (
                client.api.services.enclave.request_assets_transfer_to_enclave(
                    service_func_name=self.code.service_func_name
                )
            )
            if isinstance(assets_transferred, SyftError):
                raise SyftException(assets_transferred.message)
            print(assets_transferred.message)

        result_parts = []
        for client in self.clients.values():
            result = client.api.services.enclave.request_execution(
                service_func_name=self.code.service_func_name
            )
            if isinstance(result, SyftError):
                raise SyftException(result.message)
            result_parts.append(result)

        # TODO reconstruct the result from the parts and return
        # TODO Cleanup the Enclave in owner domain node
        return result_parts[0]

    def _get_clients_from_code(self) -> dict[UID, SyftClient]:
        if not self.code or not self.code.input_policy_init_kwargs:
            return {}

        clients = {
            policy.node_id: client
            for policy in self.code.input_policy_init_kwargs.keys()
            if
            (
                # TODO use node_uid, verify_key instead as there could be multiple logged-in users to the same client
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

    def _pre_execution_request_checks(self) -> bool:
        members_nodes_pending_approval = [
            request.syft_node_location
            for request in self.requests
            if request.status == RequestStatus.PENDING
        ]
        if members_nodes_pending_approval:
            member_names = [
                self._get_node_name(member_node_id) or f"Node ID: {member_node_id}"
                for member_node_id in members_nodes_pending_approval
            ]
            raise SyftException(
                f"Cannot execute project as approval request is pending for {human_friendly_join(member_names)}."
            )
        return True

    def _get_node_name(self, node_id: UID) -> str | None:
        node_identity = self.members.get(node_id)
        return node_identity.node_name if node_identity else None
