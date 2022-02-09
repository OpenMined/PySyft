# relative
from .permissions import BasePermission


class UserCanTriageRequest(BasePermission):
    # TODO: Exceptions could fail silently either depending upon the message
    # # or the permission (not sure which one, need to discuss)

    def has_permission(self, node, verify_key):
        return (
            node.users.can_triage_requests(verify_key=verify_key)
            if verify_key
            else False
        )


class UserCanCreateUsers(BasePermission):
    def has_permission(self, node, verify_key):
        return {
            node.users.can_create_users(verify_key=verify_key) if verify_key else False
        }


class UserCanEditRoles(BasePermission):
    def has_permission(self, node, verify_key):
        return {
            node.users.can_edit_roles(verify_key=verify_key) if verify_key else False
        }


class UserCanUploadData(BasePermission):
    def has_permission(self, node, verify_key):
        return {
            node.users.can_upload_data(verify_key=verify_key) if verify_key else False
        }
