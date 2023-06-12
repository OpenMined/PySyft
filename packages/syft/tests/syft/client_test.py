def test_client_logged_in_user(worker):
    guest_client = worker.guest_client
    assert guest_client.logged_in_user == ""

    guest_client.login(email="info@openmined.org", password="changethis")
    assert guest_client.logged_in_user == "info@openmined.org"

    guest_client.register(
        name="sheldon", email="sheldon@caltech.edu", password="bazinga"
    )

    guest_client.login(email="sheldon@caltech.edu", password="bazinga")

    assert guest_client.logged_in_user == "sheldon@caltech.edu"
