# relative
from ..permissions import BasePermission


class UserCanTriageRequest(BasePermission):
    # TODO: Exceptions could fail silently either depending upon the message
    # # or the permission (not sure which one, need to discuss)

    def has_permission(self, node, verify_key):
        _allowed = (
            node.users.can_triage_requests(verify_key=verify_key)
            if verify_key
            else False
        )
        return _allowed
