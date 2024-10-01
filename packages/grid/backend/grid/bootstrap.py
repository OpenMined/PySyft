# stdlib
import argparse
from collections.abc import Callable
import json
import os
import uuid

# third party
from nacl.encoding import HexEncoder
from nacl.signing import SigningKey

# we want to bootstrap servers with persistent uids and keys and allow a variety of ways
# to resolve these at startup

# first we check the environment variables
# then we check for a file
# then we can check the database
# we raise an Exception if the values passed in to the ENV variables dont match or
# the values from anywhere are invalid


def get_env(key: str, default: str = "") -> str | None:
    uid = str(os.environ.get(key, default))
    if len(uid) > 0:
        return uid
    return None


CREDENTIALS_PATH = str(get_env("CREDENTIALS_PATH", "credentials.json"))
SERVER_PRIVATE_KEY = "SERVER_PRIVATE_KEY"
SERVER_UID = "SERVER_UID"


def get_credentials_file() -> dict[str, str]:
    try:
        if os.path.exists(CREDENTIALS_PATH):
            with open(CREDENTIALS_PATH) as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def get_credentials_file_key(key: str) -> str | None:
    credentials = get_credentials_file()
    if key in credentials:
        return credentials[key]
    return None


def save_credential(key: str, value: str) -> str:
    credentials = get_credentials_file()
    if key in credentials:
        raise Exception(f"{key} already set in credentials file. Can't overwrite.")
    credentials[key] = value
    try:
        dirname = os.path.dirname(CREDENTIALS_PATH)
        if not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)
        with open(CREDENTIALS_PATH, "w") as f:
            f.write(f"{json.dumps(credentials)}")
    except Exception as e:
        raise e
    return value


def generate_server_uid() -> str:
    return str(uuid.uuid4())


def key_to_str(key: bytes) -> str:
    return key.encode(encoder=HexEncoder).decode("utf-8")


def generate_private_key() -> str:
    return key_to_str(SigningKey.generate())


def get_private_key_env() -> str | None:
    return get_env(SERVER_PRIVATE_KEY)


def get_server_uid_env() -> str | None:
    return get_env(SERVER_UID)


def validate_private_key(private_key: str | bytes) -> str:
    try:
        if isinstance(private_key, str):
            key = SigningKey(bytes.fromhex(private_key))
        elif isinstance(private_key, bytes):
            key = SigningKey(private_key)
        str_key = key_to_str(key)
        if str_key == private_key:
            return str_key
    except Exception:
        pass
    raise Exception(f"{SERVER_PRIVATE_KEY} is invalid")


def validate_uid(server_uid: str) -> str:
    try:
        uid = uuid.UUID(server_uid)
        if server_uid == uid.hex or server_uid == str(uid):
            return str(uid)
    except Exception:
        pass
    raise Exception(f"{SERVER_UID} is invalid")


def get_credential(
    key: str, validation_func: Callable, generation_func: Callable
) -> str:
    """
    This method will try to get a credential and if it isn't supplied or doesn't exist
    it will generate one and save it. If the one supplied doesn't match the one saved
    it will raise an Exception.
    """
    file_credential = get_credentials_file_key(key)
    env_credential = get_env(key)

    # supplying a different key means something has gone wrong so raise Exception
    if (
        file_credential is not None
        and env_credential is not None
        and validation_func(file_credential) != validation_func(env_credential)
    ):
        raise Exception(f"{key} from ENV must match {key} in {CREDENTIALS_PATH}")

    if env_credential is not None and file_credential is None:
        # if there is no key in the credential file but one is passed in ENV then save it
        return save_credential(key, validation_func(env_credential))
    elif file_credential is None and env_credential is None:
        # we can generate one for the user and save it
        return save_credential(key, validation_func(generation_func()))
    elif file_credential is not None and validation_func(file_credential):
        return file_credential

    raise Exception("Failed to get or generate Private Key")


def get_private_key() -> str:
    return get_credential(
        SERVER_PRIVATE_KEY, validate_private_key, generate_private_key
    )


def get_server_uid() -> str:
    return get_credential(SERVER_UID, validate_uid, generate_server_uid)


def delete_credential_file() -> None:
    if os.path.exists(CREDENTIALS_PATH):
        os.unlink(CREDENTIALS_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--private_key", action="store_true", help="Get Private Key")
    parser.add_argument("--uid", action="store_true", help="Get UID")
    parser.add_argument(
        "--file", action="store_true", help="Generate credentials as file"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Show ENV and file credentials"
    )
    args = parser.parse_args()

    if args.private_key or args.uid:
        if args.private_key:
            print(get_private_key())
        elif args.uid:
            print(get_server_uid())
    elif args.file:
        delete_credential_file()
        get_private_key()
        get_server_uid()
        print(f"Generated credentials file at '{CREDENTIALS_PATH}'")
    elif args.debug:
        print("Credentials File", get_credentials_file())
        print(SERVER_PRIVATE_KEY, "=", get_private_key_env())
        print(SERVER_UID, "=", get_server_uid_env())
    else:
        parser.print_help()
