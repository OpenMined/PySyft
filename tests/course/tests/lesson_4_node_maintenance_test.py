# third party
import pytest
from testbook import testbook


@pytest.fixture(scope="module")
def tb():
    with testbook("../courses/L4_NodeMaintenance.ipynb", execute=range(30)) as tb:
        yield tb


def test_login(tb):
    domain_node = tb.ref("domain_node")
    # Check if data owner client is initialized
    assert domain_node is not None
    assert domain_node.version is not None
    # Check if login messages were printed
    assert tb.cell_output_text(4) is not None
    # Check if users are present in the domain node
    expected_emails = [
        "info@openmined.org",
        "sheldon@caltech.edu",
    ]
    emails = tb.ref("list(domain_node.users.pandas()['email'].values)")
    assert set(emails) == set(expected_emails)

    # Check if data scientist client is initialized
    assert tb.ref("data_scientist_node") is not None
    assert tb.ref("data_scientist_node.version") is not None


def test_user_details_and_permissions(tb):
    # Verify user details and roles
    domain_node = tb.ref("domain_node")
    all_users = domain_node.users.all()
    for user in all_users:
        if user["email"] == "sheldon@catech.edu":
            assert user["role"] == "Data Scientist"
            assert user["budget"] == "100.0"
            assert user["name"] == "Sheldon Cooper"
        elif user["email"] == "info@openmined.org":
            assert user["role"] == "Owner"


def test_budget_requests(tb):
    data_scientist_node = tb.ref("data_scientist_node")
    assert data_scientist_node.privacy_budget == 100.0
    # A request has been raised
    assert tb.ref("data_scientist_node.requests[0]") is not None
    ds_request = tb.ref("data_scientist_node.requests[0]")
    assert ds_request.requested_budget == 1000.0
    assert ds_request.request_description == "I want to do data exploration"
    do_request = tb.ref("domain_node.requests[0]")
    # Check if request raised by Data Scientist is same as the request received by Data Owner
    assert ds_request == do_request
