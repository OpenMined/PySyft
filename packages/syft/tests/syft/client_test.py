def test_client_logged_in_user(worker):
    guest_client = worker.guest_client
    assert guest_client.logged_in_user == ""

    client = guest_client.login(email="info@openmined.org", password="changethis")
    assert client.logged_in_user == "info@openmined.org"

    client.register(
        name="sheldon",
        email="sheldon@caltech.edu",
        password="bazinga",
        password_verify="bazinga",
    )

    client = client.login(email="sheldon@caltech.edu", password="bazinga")

    assert client.logged_in_user == "sheldon@caltech.edu"
