import glob
import os.path
import json
from syft.grid.authentication.credential import AbstractCredential


class AccountCredential(AbstractCredential):
    """Parse/represent credentials based on username-password structure. 
        Expected JSON Format:
        { "accounts": [ {"user": "example1", "password": "pass_example"},
                          {"user": "user2", "password": "password2"},
                           ....
                        ]
        }
    """

    # Constants used to parse user credential files
    USERNAME_FIELD = "username"
    PASSWORD_FIELD = "password"
    CREDENTIAL_FIELD = "accounts"

    def __init__(self, username, password):
        """ Initialize a user authentication object.
            Args:
                username (str) : Key to identify this object.
                password (str) : Secret used to validate user.
        """
        self.username = username
        self.password = password
        super().__init__()

    @staticmethod
    def parse(path: str, file_name: str):
        """ Static method used to create new account authentication instances
            parsing a json file.
            Args:
                path (str) : Json file path.
                file_name (str) : File's name.
            Returns:
                List : List of account objects.
        """
        user_files = glob.glob(os.path.join(path, file_name))
        users = dict()
        for f in user_files:
            with open(f) as json_file:
                credentials = json.load(json_file)
                cred_users = credentials[AccountCredential.CREDENTIAL_FIELD]
                for user in cred_users:
                    new_user = AccountCredential(
                        user[AccountCredential.USERNAME_FIELD],
                        user[AccountCredential.PASSWORD_FIELD],
                    )
                    users[new_user.username] = new_user
        return users

    def json(self):
        """  Convert account instances into a JSON/Dictionary structure. """
        return {
            AccountCredential.USERNAME_FIELD: self.username,
            AccountCredential.PASSWORD_FIELD: self.password,
        }

    def __str__(self):
        return (
            "< AccountCredential - User: "
            + self.username
            + " Password: "
            + ("*" * len(self.password))
            + " >"
        )
