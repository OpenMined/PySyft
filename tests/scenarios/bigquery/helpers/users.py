# stdlib
from dataclasses import dataclass
from typing import Any

# third party
from faker import Faker
from unsync import unsync

# syft absolute
import syft as sy
from syft.service.user.user_roles import ServiceRole


@dataclass
class TestUser:
    name: str
    email: str
    password: str
    role: ServiceRole
    server_cache: Any | None = None

    def client(self, server=None):
        if server is None:
            server = self.server_cache
        else:
            self.server_cache = server

        return server.login(email=self.email, password=self.password)


@unsync
async def set_settings_allow_guest_signup(
    events, client, enabled, event_name: str | None = None
):
    result = client.settings.allow_guest_signup(enable=enabled)
    if event_name:
        if isinstance(result, sy.SyftSuccess):
            events.register(event_name)


@unsync
async def check_users_created(events, client, users, event_name, event_set):
    expected_emails = {user.email for user in users}
    found_emails = set()
    await events.await_for(event_name=event_name)
    user_results = client.api.services.user.get_all()
    for user_result in user_results:
        if user_result.email in expected_emails:
            found_emails.add(user_result.email)

    if len(found_emails) == len(expected_emails):
        events.register(event_set)


def guest_register(client, test_user):
    guest_client = client.guest()
    fake = Faker()
    result = guest_client.register(
        name=test_user.name,
        email=test_user.email,
        password=test_user.password,
        password_verify=test_user.password,
        institution=fake.company(),
        website=fake.url(),
    )
    return result
