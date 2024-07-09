# stdlib
import secrets

# syft absolute
from syft.service.action.action_permissions import ActionObjectPermission
from syft.service.action.action_permissions import ActionPermission
from syft.service.action.action_permissions import COMPOUND_ACTION_PERMISSION
from syft.service.action.action_permissions import SyftVerifyKey
from syft.service.action.action_permissions import UID


def test_permission_string_round_trip():
    for permission in ActionPermission:
        uid = UID()
        if permission in COMPOUND_ACTION_PERMISSION:
            verify_key = None
        else:
            verify_key = SyftVerifyKey.from_string(secrets.token_hex(32))

        original_obj = ActionObjectPermission(uid, permission, verify_key)
        perm_string = original_obj.permission_string
        recreated_obj = ActionObjectPermission.from_permission_string(uid, perm_string)

        assert original_obj.permission == recreated_obj.permission
        assert original_obj.uid == recreated_obj.uid
        assert original_obj.credentials == recreated_obj.credentials
