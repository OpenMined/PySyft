# stdlib
from dataclasses import dataclass
from dataclasses import field
import json
import re
from typing import Any

# third party
from faker import Faker

# syft absolute
from syft.service.user.user_roles import ServiceRole

fake = Faker()


@dataclass
class TestUser:
    name: str
    email: str
    password: str
    role: ServiceRole
    new_password: str | None = None
    email_disabled: bool = False
    reset_token: str | None = None
    _client_cache: Any | None = field(default=None, repr=False, init=False)

    @property
    def latest_password(self) -> str:
        if self.new_password:
            return self.new_password
        return self.password

    def make_new_password(self) -> str:
        self.new_password = fake.password()
        return self.new_password

    @property
    def client(self):
        return self._client_cache

    def relogin(self) -> None:
        self.client = self.client

    @client.setter
    def client(self, client):
        client = client.login(email=self.email, password=self.latest_password)
        self._client_cache = client

    def to_dict(self) -> dict:
        output = {}
        for k, v in self.__dict__.items():
            if k.startswith("_"):
                continue
            if k == "role":
                v = str(v)
            output[k] = v
        return output

    def __iter__(self):
        for key, val in self.to_dict().items():
            if key.startswith("_"):
                yield val

    def __getitem__(self, key):
        if key.startswith("_"):
            return None
        return self.to_dict()[key]


def save_users(users):
    user_dicts = []
    for user in users:
        user_dicts.append(user.to_dict())
    print(user_dicts)
    with open("./users.json", "w") as f:
        f.write(json.dumps(user_dicts))


def load_users(path="./users.json"):
    with open(path) as f:
        data = f.read()
        user_dicts = json.loads(data)
    return [TestUser(**user) for user in user_dicts]


def make_user(
    name: str | None = None,
    email: str | None = None,
    password: str | None = None,
    role: ServiceRole = ServiceRole.DATA_SCIENTIST,
):
    fake = Faker()
    if name is None:
        name = fake.name()
    if email is None:
        ascii_string = re.sub(r"[^a-zA-Z\s]", "", name).lower()
        dashed_string = ascii_string.replace(" ", "-")
        email = f"{dashed_string}-fake@openmined.org"
    if password is None:
        password = fake.password()

    return TestUser(name=name, email=email, password=password, role=role)


def user_exists(root_client, email: str) -> bool:
    users = root_client.api.services.user
    for user in users:
        if user.email == email:
            return True
    return False


def create_user(root_client, test_user):
    if not user_exists(root_client, test_user.email):
        fake = Faker()
        root_client.register(
            name=test_user.name,
            email=test_user.email,
            password=test_user.password,
            password_verify=test_user.password,
            institution=fake.company(),
            website=fake.url(),
        )
    else:
        print("User already exists", test_user)
