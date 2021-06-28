# third party
from bcrypt import checkpw
import pytest
from src.main.core.database import *
from src.main.core.exceptions import RoleNotFoundError, InvalidCredentialsError
from src.main.core.manager.role_manager import RoleManager

user_role = ("User", False, False, False, False, False, False, False)
admin_role = ("Administrator", True, True, True, True, False, False, True)
owner_role = ("Owner", True, True, True, True, True, True, True)
officer_role = ("Compliance Officer", True, False, False, False, False, False, False)


@pytest.fixture
def cleanup(database):
    yield
    try:
        database.session.query(User).delete()
        database.session.query(Role).delete()
        database.session.query(Group).delete()
        database.session.query(UserGroup).delete()
        database.session.commit()
    except:
        database.session.rollback()


def test_create_role_manager(database, cleanup):
    users = RoleManager(database)


def test_user_role(database, cleanup):
    role_manager = RoleManager(database)
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    database.session.commit()

    retrieved_role = role_manager.user_role

    assert retrieved_role.id == 3
    assert retrieved_role.name == "User"
    assert retrieved_role.can_edit_settings == False
    assert retrieved_role.can_create_groups == False
    assert retrieved_role.can_manage_infrastructure == False
    assert retrieved_role.can_upload_data == False
    assert retrieved_role.can_create_users == False
    assert retrieved_role.can_triage_requests == False
    assert retrieved_role.can_edit_roles == False


def test_user_role_mutiple_roles(database, cleanup):
    role_manager = RoleManager(database)
    new_role_1 = create_role(*user_role)
    new_role_2 = create_role(*user_role)
    new_role_3 = create_role(*user_role)
    database.session.add(new_role_1)
    database.session.add(new_role_2)
    database.session.add(new_role_3)
    database.session.commit()

    database.session.delete(new_role_1)
    database.session.commit()

    retrieved_role = role_manager.user_role

    assert retrieved_role.id == 2
    assert retrieved_role.name == "User"
    assert retrieved_role.can_edit_settings == False
    assert retrieved_role.can_create_groups == False
    assert retrieved_role.can_manage_infrastructure == False
    assert retrieved_role.can_upload_data == False
    assert retrieved_role.can_create_users == False
    assert retrieved_role.can_triage_requests == False
    assert retrieved_role.can_edit_roles == False


def test_admin_role(database, cleanup):
    role_manager = RoleManager(database)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    database.session.commit()

    retrieved_role = role_manager.admin_role

    assert retrieved_role.id == 3
    assert retrieved_role.name == "Administrator"
    assert retrieved_role.can_edit_settings == True
    assert retrieved_role.can_create_groups == True
    assert retrieved_role.can_manage_infrastructure == False
    assert retrieved_role.can_upload_data == True
    assert retrieved_role.can_create_users == True
    assert retrieved_role.can_triage_requests == True
    assert retrieved_role.can_edit_roles == False


def test_owner_role(database, cleanup):
    role_manager = RoleManager(database)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_role = create_role(*owner_role)
    database.session.add(new_role)
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    database.session.commit()
    new_role = create_role(*owner_role)
    database.session.add(new_role)

    retrieved_role = role_manager.owner_role

    assert retrieved_role.id == 3
    assert retrieved_role.name == "Owner"
    assert retrieved_role.can_edit_settings == True
    assert retrieved_role.can_create_groups == True
    assert retrieved_role.can_manage_infrastructure == True
    assert retrieved_role.can_upload_data == True
    assert retrieved_role.can_create_users == True
    assert retrieved_role.can_triage_requests == True
    assert retrieved_role.can_edit_roles == True


def test_compliance_officer_role(database, cleanup):
    role_manager = RoleManager(database)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_role = create_role(*owner_role)
    database.session.add(new_role)
    new_role = create_role(*officer_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    database.session.commit()
    new_role = create_role(*officer_role)
    database.session.add(new_role)

    retrieved_role = role_manager.compliance_officer_role

    assert retrieved_role.id == 3
    assert retrieved_role.name == "Compliance Officer"
    assert retrieved_role.can_edit_settings == False
    assert retrieved_role.can_create_groups == False
    assert retrieved_role.can_manage_infrastructure == False
    assert retrieved_role.can_upload_data == False
    assert retrieved_role.can_create_users == False
    assert retrieved_role.can_triage_requests == True
    assert retrieved_role.can_edit_roles == False


def test_common_roles(database, cleanup):
    role_manager = RoleManager(database)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_role = create_role(*owner_role)
    database.session.add(new_role)
    new_role = create_role(*officer_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*officer_role)
    database.session.add(new_role)
    database.session.commit()

    common_roles = role_manager.common_roles

    assert len(common_roles) == 2

    assert common_roles[0].id == 1
    assert common_roles[0].name == "User"
    assert common_roles[0].can_edit_settings == False
    assert common_roles[0].can_create_groups == False
    assert common_roles[0].can_manage_infrastructure == False
    assert common_roles[0].can_upload_data == False
    assert common_roles[0].can_create_users == False
    assert common_roles[0].can_triage_requests == False
    assert common_roles[0].can_edit_roles == False

    assert common_roles[1].id == 4
    assert common_roles[1].name == "User"
    assert common_roles[1].can_edit_settings == False
    assert common_roles[1].can_create_groups == False
    assert common_roles[1].can_manage_infrastructure == False
    assert common_roles[1].can_upload_data == False
    assert common_roles[1].can_create_users == False
    assert common_roles[1].can_triage_requests == False
    assert common_roles[1].can_edit_roles == False


def test_org_roles(database, cleanup):
    role_manager = RoleManager(database)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_role = create_role(*owner_role)
    database.session.add(new_role)
    new_role = create_role(*officer_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    database.session.commit()

    org_roles = role_manager.org_roles

    assert len(org_roles) == 3

    assert org_roles[0].id == 2
    assert org_roles[0].name == "Owner"
    assert org_roles[0].can_edit_settings == True
    assert org_roles[0].can_create_groups == True
    assert org_roles[0].can_manage_infrastructure == True
    assert org_roles[0].can_upload_data == True
    assert org_roles[0].can_create_users == True
    assert org_roles[0].can_triage_requests == True
    assert org_roles[0].can_edit_roles == True

    assert org_roles[1].id == 3
    assert org_roles[1].name == "Compliance Officer"
    assert org_roles[1].can_edit_settings == False
    assert org_roles[1].can_create_groups == False
    assert org_roles[1].can_manage_infrastructure == False
    assert org_roles[1].can_upload_data == False
    assert org_roles[1].can_create_users == False
    assert org_roles[1].can_triage_requests == True
    assert org_roles[1].can_edit_roles == False

    assert org_roles[2].id == 5
    assert org_roles[2].name == "Administrator"
    assert org_roles[2].can_edit_settings == True
    assert org_roles[2].can_create_groups == True
    assert org_roles[2].can_manage_infrastructure == False
    assert org_roles[2].can_upload_data == True
    assert org_roles[2].can_create_users == True
    assert org_roles[2].can_triage_requests == True
    assert org_roles[2].can_edit_roles == False


def test_first(database, cleanup):
    role_manager = RoleManager(database)
    new_role = create_role(*owner_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_role = create_role(*officer_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    database.session.commit()

    retrieved_role = role_manager.first(**{"name": "User"})

    assert retrieved_role.id == 2
    assert retrieved_role.name == "User"
    assert retrieved_role.can_edit_settings == False
    assert retrieved_role.can_create_groups == False
    assert retrieved_role.can_manage_infrastructure == False
    assert retrieved_role.can_upload_data == False
    assert retrieved_role.can_create_users == False
    assert retrieved_role.can_triage_requests == False
    assert retrieved_role.can_edit_roles == False


def test_first_fail(database, cleanup):
    role_manager = RoleManager(database)
    new_role = create_role(*owner_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_role = create_role(*officer_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    database.session.commit()

    with pytest.raises(RoleNotFoundError):
        retrieved_role = role_manager.first(**{"name": "Invalid"})


def test_query(database, cleanup):
    role_manager = RoleManager(database)
    new_role = create_role(*owner_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_role = create_role(*officer_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    database.session.commit()

    retrieved_roles = role_manager.query(name="Compliance Officer")

    assert len(retrieved_roles) == 1
    assert retrieved_roles[0].id == 3
    assert retrieved_roles[0].name == "Compliance Officer"
    assert retrieved_roles[0].can_edit_settings == False
    assert retrieved_roles[0].can_create_groups == False
    assert retrieved_roles[0].can_manage_infrastructure == False
    assert retrieved_roles[0].can_upload_data == False
    assert retrieved_roles[0].can_create_users == False
    assert retrieved_roles[0].can_triage_requests == True
    assert retrieved_roles[0].can_edit_roles == False


def test_query_fail(database, cleanup):
    role_manager = RoleManager(database)
    new_role = create_role(*owner_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_role = create_role(*officer_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    database.session.commit()

    with pytest.raises(RoleNotFoundError):
        retrieved_roles = role_manager.query(name="404 Officer")


def test_set(database, cleanup):
    role_manager = RoleManager(database)
    new_role = create_role(*owner_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_role = create_role(*officer_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    database.session.commit()

    role_manager.set(3, {"name": "404 Officer", "can_upload_data": True})

    edited_role = database.session.query(Role).get(3)

    assert edited_role.name == "404 Officer"
    assert edited_role.can_edit_settings == False
    assert edited_role.can_create_groups == False
    assert edited_role.can_manage_infrastructure == False
    assert edited_role.can_upload_data == True
    assert edited_role.can_create_users == False
    assert edited_role.can_triage_requests == True
    assert edited_role.can_edit_roles == False


def test_set_fail(database, cleanup):
    role_manager = RoleManager(database)
    new_role = create_role(*owner_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_role = create_role(*officer_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    database.session.commit()

    with pytest.raises(RoleNotFoundError):
        role_manager.set(10, {"name": "404 Officer", "can_upload_data": True})
