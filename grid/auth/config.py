import os.path
from . import BASE_DIR, BASE_FOLDER, AUTH_MODELS
from . import auth_credentials
from . import UserAuthentication
import getpass
import json


def register_new_credentials(path: str) -> UserAuthentication:
    """ Create a new credential if not found any credential file during load_credentials function.

        Args:
            path (str) : File path.
        Returns:
            user (UserAuthentication) : New credential instance.
    """
    # Create a new credential
    username = input(UserAuthentication.USERNAME_FIELD + ": ")
    password = getpass.getpass(UserAuthentication.PASSWORD_FIELD + ": ")
    first_user = {
        UserAuthentication.USERNAME_FIELD: username,
        UserAuthentication.PASSWORD_FIELD: password,
    }
    credentials = json.dumps({"credential": [first_user]})

    # Save at BASE_DIR/BASE_FOLDER/UserAuthentication.FILENAME (JSON format)
    file_path = os.path.join(path, UserAuthentication.FILENAME)
    auth_file = open(file_path, "w")
    auth_file.write(credentials)
    auth_file.close()

    return UserAuthentication(username, password)


def read_authentication_configs(directory=None, folder=None) -> list:
    """ Search for a path and folder used to store user credentials
        
        Args:
            directory (str) : System path (can usually be /home/<user>).
            folder (str) : folder name used to store PyGrid credentials.

        Returns:
            List : List of credentials instances.
    """
    dir_path = directory if directory else BASE_DIR
    folder_name = folder if folder else BASE_FOLDER

    path = os.path.join(dir_path, folder_name)

    # IF directory aready exists.
    if os.path.isdir(path):
        # Check / parse every credential files.
        # Initialize authentication objects.
        # Save Objects at auth_credentials list
        for model in AUTH_MODELS:
            auth_credentials.extend(model.parse(path))
    else:
        # Create Base DIR
        os.mkdir(path)

    # If auth_credentials is empty
    if not len(auth_credentials):
        # Create new one
        auth_credentials.append(register_new_credentials(path))
    return auth_credentials


def search_credential(user: str):
    """ Search for a specific credential instance.
        
        Args:
            user (str) : Key used to identify the credential.
        Returns:
            BaseAuthentication : Credential's instance.
    """
    for cred in auth_credentials:
        if cred.username == user:
            return cred
