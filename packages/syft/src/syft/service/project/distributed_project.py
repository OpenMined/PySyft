# stdlib
from typing import Any
from typing import Self

# third party
from pydantic import field_validator

# relative
from ...client.api import NodeIdentity
from ...client.client import SyftClient
from ...client.client import SyftClientSessionCache
from ...serde.serializable import serializable
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SyftObject
from ...types.uid import UID
from ...util import options
from ...util.colors import SURFACE
from ..code.user_code import SubmitUserCode
from ..metadata.node_metadata import NodeMetadata
from ..request.request import Request
from ..request.request import RequestStatus
from ..response import SyftError
from ..response import SyftException
from .project import Project
from .project import ProjectRequest
from .project import ProjectSubmit


@serializable(without=["clients"])
class DistributedProject(SyftObject):
    __canonical_name__ = "DistributedProject"
    __version__ = SYFT_OBJECT_VERSION_1

    __hash_exclude_attrs__ = ["clients"]

    # stash rules
    __repr_attrs__ = ["name", "description", "created_by"]
    __attr_unique__ = ["name"]

    id: UID
    name: str
    description: str = ""
    code: SubmitUserCode | None = None  # only one code per project for this prototype
    clients: list[SyftClient] | None = None  # List of member clients, not persisted
    members: list[NodeIdentity] | None = (
        None  # List of member client identities, persisted
    )
    created_by: str | None = None
    username: str | None = None
    all_projects: list[Project] = []
    project_permissions: set[str] = set()  # Unused at the moment

    def _coll_repr_(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "created by": self.created_by,
            "pending requests": self.pending_requests,
        }

    def _repr_html_(self) -> Any:
        return (
            f"""
            <style>
            .syft-project {{color: {SURFACE[options.color_theme]};}}
            </style>
            """
            + "<div class='syft-project'>"
            + f"<h3>{self.name}</h3>"
            + f"<p>{self.description}</p>"
            + f"<p><strong>Created by: </strong>{self.username} ({self.created_by})</p>"
            + self.requests._repr_html_()
            + "</div>"
        )

    def __init__(self, *args: Any, **kwargs: Any):
        code = kwargs.pop("code", None)
        super().__init__(*args, **kwargs)
        if code:
            self.add_code(code)

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

    def add_code(self, code: SubmitUserCode) -> None:
        self.code = code
        self.clients = self._get_clients_from_code(code)
        self.members = list(map(self._to_node_identity, self.clients))
        self.created_by = self.clients[0].logged_in_user
        self.username = self.clients[0].logged_in_username

    def send(self) -> Self:
        if not self.code:
            raise SyftException("Cannot send project without code.")
        self._pre_submit_checks(self.clients)
        projects_map = self._send_project_to_all_clients(self.clients)
        self.all_projects = list(projects_map.values())
        return self

    @staticmethod
    def _get_clients_from_code(code: SubmitUserCode) -> list[SyftClient]:
        if not code.input_policy_init_kwargs:
            return []

        clients = [
            client
            for policy in code.input_policy_init_kwargs.keys()
            if (
                client := SyftClientSessionCache.get_client_for_node_uid(policy.node_id)
            )
        ]
        return clients

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

    @staticmethod
    def _to_node_identity(val: SyftClient) -> NodeIdentity:
        if isinstance(val, SyftClient) and val.metadata is not None:
            metadata = val.metadata.to(NodeMetadata)
            return metadata.to(NodeIdentity)
        else:
            raise SyftException(f"members must be SyftClient. Received: {type(val)}")

    def _pre_submit_checks(self, clients: list[SyftClient]) -> bool:
        try:
            # Check if the user can create projects
            for client in clients:
                result = client.api.services.project.can_create_project()
                if isinstance(result, SyftError):
                    raise SyftException(result.message)
        except Exception:
            raise SyftException("Only Data Scientists can create projects")

        return True

    def _send_project_to_all_clients(
        self, clients: list[SyftClient]
    ) -> dict[SyftClient, Project]:
        projects_map: dict[SyftClient, Project] = {}
        for client in clients:
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
