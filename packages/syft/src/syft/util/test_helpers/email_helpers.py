# stdlib
from dataclasses import dataclass
from dataclasses import field
import json
import re
import time
from typing import Any

# third party
from aiosmtpd.controller import Controller
from faker import Faker
from filelock import FileLock

# relative
from ...service.user.user_roles import ServiceRole

fake = Faker()


@dataclass
class Email:
    email_from: str
    email_to: str
    email_content: str

    def to_dict(self) -> dict:
        output = {}
        for k, v in self.__dict__.items():
            output[k] = v
        return output

    def __iter__(self):
        yield from self.to_dict().items()

    def __getitem__(self, key):
        return self.to_dict()[key]

    def __repr__(self) -> str:
        return f"{self.email_to}\n{self.email_from}\n\n{self.email_content}"


class EmailServer:
    def __init__(self, filepath="emails.json"):
        self.filepath = filepath
        lockpath = self.filepath + ".lock"
        self._lock = FileLock(lock_file=lockpath)
        self._emails: dict[str, list[Email]] = self.load_emails()

    def load_emails(self) -> dict[str, list[Email]]:
        try:
            with (
                self._lock as _,
                open(self.filepath) as f,
            ):
                data = json.load(f)
                return {k: [Email(**email) for email in v] for k, v in data.items()}
        except Exception as e:
            print("Issues reading email file. Using empty email dict.", e)
            return {}

    def save_emails(self) -> None:
        with (
            self._lock as _,
            open(self.filepath, "w") as f,
        ):
            data = {
                k: [email.to_dict() for email in v] for k, v in self._emails.items()
            }
            f.write(json.dumps(data))

    def add_email_for_user(self, user_email: str, email: Email) -> None:
        with self._lock:
            if user_email not in self._emails:
                self._emails[user_email] = []
            self._emails[user_email].append(email)
            self.save_emails()

    def get_emails_for_user(self, user_email: str) -> list[Email]:
        self._emails: dict[str, list[Email]] = self.load_emails()
        return self._emails.get(user_email, [])

    def reset_emails(self) -> None:
        with self._lock:
            self._emails = {}
            self.save_emails()


SENDER = "noreply@openmined.org"


def get_token(email) -> str:
    # stdlib
    import re

    pattern = r"syft_client\.reset_password\(token='(.*?)', new_password=.*?\)"
    try:
        token = re.search(pattern, email.email_content).group(1)
    except Exception:
        raise Exception(f"No token found in email: {email.email_content}")
    return token


@dataclass
class TestUser:
    name: str
    email: str
    password: str
    role: ServiceRole
    new_password: str | None = None
    email_disabled: bool = False
    reset_password: bool = False
    reset_token: str | None = None
    _client_cache: Any | None = field(default=None, repr=False, init=False)
    _email_server: EmailServer | None = None

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
        this_client = client.login(email=self.email, password=self.latest_password)
        self._client_cache = this_client

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
            if not key.startswith("_"):
                yield key, val

    def __getitem__(self, key):
        if key.startswith("_"):
            return None
        return self.to_dict()[key]

    def update_password(self):
        self.password = self.new_password
        self.new_password = None

    @property
    def emails(self) -> list[Email]:
        if not self._email_server:
            print("Not connected to email server object")
            return []
        return self._email_server.get_emails_for_user(self.email)

    def get_token(self) -> str:
        for email in reversed(self.emails):
            token = None
            try:
                token = get_token(email)
                break
            except Exception:  # nosec
                pass
        self.reset_token = token
        return token


def save_users(users):
    user_dicts = []
    for user in users:
        user_dicts.append(user.to_dict())
    print(user_dicts)
    with open("./users.json", "w") as f:
        f.write(json.dumps(user_dicts))


def load_users(high_client: None, path="./users.json"):
    users = []
    with open(path) as f:
        data = f.read()
        user_dicts = json.loads(data)
    for user in user_dicts:
        test_user = TestUser(**user)
        if high_client:
            test_user.client = high_client
        users.append(test_user)
    return users


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


class SMTPTestServer:
    def __init__(self, email_server, port=9025, ready_timeout=5):
        self.port = port
        self.hostname = "0.0.0.0"  # nosec: B104
        self.controller = None

        # Simple email handler class
        class SimpleHandler:
            async def handle_DATA(self, server, session, envelope):
                try:
                    print(f"> SMTPTestServer got an email for {envelope.rcpt_tos}")
                    email = Email(
                        email_from=envelope.mail_from,
                        email_to=envelope.rcpt_tos,
                        email_content=envelope.content.decode(
                            "utf-8", errors="replace"
                        ),
                    )
                    email_server.add_email_for_user(envelope.rcpt_tos[0], email)
                    email_server.save_emails()
                    return "250 Message accepted for delivery"
                except Exception as e:
                    print(f"> Error handling email: {e}")
                    return "550 Internal Server Error"

        try:
            self.handler = SimpleHandler()
            self.controller = Controller(
                self.handler,
                hostname=self.hostname,
                port=self.port,
                ready_timeout=ready_timeout,
            )
        except Exception as e:
            print(f"> Error initializing SMTPTestServer Controller: {e}")

    def start(self):
        self.controller.start()

    def stop(self):
        self.controller.stop()

    def __del__(self):
        if self.controller:
            self.stop()


class TimeoutError(Exception):
    pass


class Timeout:
    def __init__(self, timeout_duration):
        if timeout_duration > 60:
            raise ValueError("Timeout duration cannot exceed 60 seconds.")
        self.timeout_duration = timeout_duration

    def run_with_timeout(self, condition_func, *args, **kwargs):
        start_time = time.time()
        result = None

        while True:
            elapsed_time = time.time() - start_time
            if elapsed_time > self.timeout_duration:
                raise TimeoutError(
                    f"Function execution exceeded {self.timeout_duration} seconds."
                )

            # Check if the condition is met
            try:
                if condition_func():
                    print("Condition met, exiting early.")
                    break
            except Exception as e:
                print(f"Exception in target function: {e}")
                break  # Exit the loop if an exception occurs in the function
            time.sleep(1)

        return result


def get_email_server(reset=False, port=9025):
    email_server = EmailServer()
    if reset:
        email_server.reset_emails()
    for _ in range(5):
        try:
            smtp_server = SMTPTestServer(email_server, port=port)
            smtp_server.start()
            return email_server, smtp_server

        except TimeoutError:
            del smtp_server
            print("SMTP server timed out. Retrying...")
            continue
        except Exception as e:
            print(f"> Error starting SMTP server: {e}")
    raise Exception("Failed to start SMTP server in 5 attempts.")


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
