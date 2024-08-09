# stdlib
from typing import TYPE_CHECKING

# third party
import sqlalchemy as sa
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column

# syft absolute
import syft as sy
from syft.service.code_history.code_history import CodeHistory
from syft.service.project.project import Project

# relative
from ...server.credentials import SyftVerifyKey
from ...types.uid import UID
from ..job.base_sql import Base, VerifyKeyTypeDecorator
from ..job.base_sql import CommonMixin
from ..job.base_sql import PermissionMixin
from ..job.base_sql import UIDTypeDecorator


class ProjectDB(CommonMixin, Base, PermissionMixin):
    __tablename__ = "projects"

    name: Mapped[str]
    description: Mapped[str | None]
    members: Mapped[bytes]
    users: Mapped[bytes]
    username: Mapped[str | None]
    created_by: Mapped[str]
    start_hash: Mapped[str | None]
    user_signing_key: Mapped[bytes | None]
    events: Mapped[bytes]
    event_id_hashmap: Mapped[bytes]
    state_sync_leader: Mapped[bytes]
    leader_server_peer: Mapped[bytes | None]
    consensus_model: Mapped[bytes]
    project_permissions: Mapped[bytes]

    @classmethod
    def from_obj(cls, obj: Project) -> "ProjectDB":
        return cls(
            id=obj.id,
            name=obj.name,
            description=obj.description,
            members=sy.serialize(obj.members, to_bytes=True),
            users=sy.serialize(obj.users, to_bytes=True),
            username=obj.username,
            created_by=obj.created_by,
            start_hash=obj.start_hash,
            user_signing_key=sy.serialize(obj.user_signing_key, to_bytes=True),
            events=sy.serialize(obj.events, to_bytes=True),
            event_id_hashmap=sy.serialize(obj.event_id_hashmap, to_bytes=True),
            state_sync_leader=sy.serialize(obj.state_sync_leader, to_bytes=True),
            leader_server_peer=sy.serialize(obj.leader_server_peer, to_bytes=True),
            consensus_model=sy.serialize(obj.consensus_model, to_bytes=True),
            project_permissions=sy.serialize(obj.project_permissions, to_bytes=True),
        )

    def to_obj(self) -> Project:
        return Project(
            id=self.id,
            name=self.name,
            description=self.description,
            members=sy.deserialize(self.members, from_bytes=True),
            users=sy.deserialize(self.users, from_bytes=True),
            username=self.username,
            created_by=self.created_by,
            start_hash=self.start_hash,
            user_signing_key=sy.deserialize(self.user_signing_key, from_bytes=True),
            events=sy.deserialize(self.events, from_bytes=True),
            event_id_hashmap=sy.deserialize(self.event_id_hashmap, from_bytes=True),
            state_sync_leader=sy.deserialize(self.state_sync_leader, from_bytes=True),
            leader_server_peer=sy.deserialize(self.leader_server_peer, from_bytes=True),
            consensus_model=sy.deserialize(self.consensus_model, from_bytes=True),
            project_permissions=sy.deserialize(
                self.project_permissions, from_bytes=True
            ),
        )


ProjectDB._init_perms()
