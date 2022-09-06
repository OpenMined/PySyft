# stdlib
from typing import Dict
from typing import List
from typing import Optional

# third party
from typing_extensions import final

# relative
from .....common.message import ImmediateSyftMessageWithReply
from .....common.message import ImmediateSyftMessageWithoutReply
from .....common.serde.serializable import serializable
from .....common.uid import UID
from .....io.address import Address


@serializable(recursive_serde=True)
@final
class CreateRoleMessage(ImmediateSyftMessageWithReply):
    __attr_allowlist__ = [
        "id",
        "address",
        "name",
        "can_make_data_requests",
        "can_triage_data_requests",
        "can_manage_privacy_budget",
        "can_create_users",
        "can_manage_users",
        "can_edit_roles",
        "can_manage_infrastructure",
        "can_upload_data",
        "can_upload_legal_document",
        "can_edit_domain_settings",
        "reply_to",
    ]

    def __init__(
        self,
        address: Address,
        name: str,
        reply_to: Address,
        can_make_data_requests: bool = False,
        can_triage_data_requests: bool = False,
        can_manage_privacy_budget: bool = False,
        can_create_users: bool = False,
        can_manage_users: bool = False,
        can_edit_roles: bool = False,
        can_manage_infrastructure: bool = False,
        can_upload_data: bool = False,
        can_upload_legal_document: bool = False,
        can_edit_domain_settings: bool = False,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)
        self.name = name

        self.can_make_data_requests = can_make_data_requests
        self.can_triage_data_requests = can_triage_data_requests
        self.can_manage_privacy_budget = can_manage_privacy_budget
        self.can_create_users = can_create_users
        self.can_manage_users = can_manage_users
        self.can_edit_roles = can_edit_roles
        self.can_manage_infrastructure = can_manage_infrastructure
        self.can_upload_data = can_upload_data
        self.can_upload_legal_document = can_upload_legal_document
        self.can_edit_domain_settings = can_edit_domain_settings


@serializable(recursive_serde=True)
@final
class GetRoleMessage(ImmediateSyftMessageWithReply):
    __attr_allowlist__ = ["id", "address", "role_id", "reply_to"]

    def __init__(
        self,
        address: Address,
        role_id: int,
        reply_to: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)
        self.role_id = role_id


@serializable(recursive_serde=True)
@final
class GetRoleResponse(ImmediateSyftMessageWithoutReply):
    __attr_allowlist__ = ["id", "address", "content"]

    def __init__(
        self,
        address: Address,
        content: Dict,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.content = content


@serializable(recursive_serde=True)
@final
class GetRolesMessage(ImmediateSyftMessageWithReply):
    __attr_allowlist__ = ["id", "address", "reply_to"]

    def __init__(
        self,
        address: Address,
        reply_to: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)


@serializable(recursive_serde=True)
@final
class GetRolesResponse(ImmediateSyftMessageWithoutReply):
    __attr_allowlist__ = ["id", "address", "content"]

    def __init__(
        self,
        address: Address,
        content: List[Dict],
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.content = content


@serializable(recursive_serde=True)
@final
class UpdateRoleMessage(ImmediateSyftMessageWithReply):
    __attr_allowlist__ = [
        "id",
        "address",
        "name",
        "can_make_data_requests",
        "can_triage_data_requests",
        "can_manage_privacy_budget",
        "can_create_users",
        "can_manage_users",
        "can_edit_roles",
        "can_manage_infrastructure",
        "can_upload_data",
        "can_upload_legal_document",
        "can_edit_domain_settings",
        "role_id",
        "reply_to",
    ]

    def __init__(
        self,
        address: Address,
        role_id: int,
        name: str,
        reply_to: Address,
        can_make_data_requests: bool = False,
        can_triage_data_requests: bool = False,
        can_manage_privacy_budget: bool = False,
        can_create_users: bool = False,
        can_manage_users: bool = False,
        can_edit_roles: bool = False,
        can_manage_infrastructure: bool = False,
        can_upload_data: bool = False,
        can_upload_legal_document: bool = False,
        can_edit_domain_settings: bool = False,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)
        self.name = name
        self.can_make_data_requests = can_make_data_requests
        self.can_triage_data_requests = can_triage_data_requests
        self.can_manage_privacy_budget = can_manage_privacy_budget
        self.can_create_users = can_create_users
        self.can_manage_users = can_manage_users
        self.can_edit_roles = can_edit_roles
        self.can_manage_infrastructure = can_manage_infrastructure
        self.can_upload_data = can_upload_data
        self.can_upload_legal_document = can_upload_legal_document
        self.can_edit_domain_settings = can_edit_domain_settings
        self.role_id = role_id


@serializable(recursive_serde=True)
@final
class DeleteRoleMessage(ImmediateSyftMessageWithReply):
    __attr_allowlist__ = ["id", "address", "reply_to", "role_id"]

    def __init__(
        self,
        address: Address,
        role_id: int,
        reply_to: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)
        self.role_id = role_id
