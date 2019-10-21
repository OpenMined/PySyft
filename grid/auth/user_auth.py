import uuid
import glob
import os.path
import json
from .authentication import BaseAuthentication


class UserAuthentication(BaseAuthentication):
    FILENAME = "auth.user"
    USERNAME_FIELD = "username"
    PASSWORD_FIELD = "password"

    def __init__(self, username, password):
        """ Initialize a user authentication object.
            Args:
                username (str) : Key to identify this object.
                password (str) : Secret used to verify and validate this object.
        """
        self.username = username
        self.password = password
        super().__init__(UserAuthentication.FILENAME)

    @staticmethod
    def parse(path):
        """ Static method used to create new user authentication instances parsing a json file.
            
            Args:
                path (str) : json file path.
            Returns:
                List : List of user authentication objects.
        """
        user_files = glob.glob(os.path.join(path, UserAuthentication.FILENAME))
        users = []
        for f in user_files:
            with open(f) as json_file:
                credentials = json.load(json_file)
                cred_users = credentials["credential"]
                for user in cred_users:
                    new_user = UserAuthentication(
                        user[UserAuthentication.USERNAME_FIELD],
                        user[UserAuthentication.PASSWORD_FIELD],
                    )
                    users.append(new_user)
        return users

    def json(self):
        """  Convert user instances into a JSON/Dictionary structure. """
        return {
            UserAuthentication.USERNAME_FIELD: self.username,
            UserAuthentication.PASSWORD_FIELD: self.password,
        }
