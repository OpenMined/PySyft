from syft.grid.authentication.account import AccountCredential
from .user_session import UserSession
from flask_login import UserMixin


class SessionsRepository(object):
    """ Sessions Repository manages user credentials and different user sessions during websocket/http requests """

    DEFAULT_MANAGER_USER = "admin"
    DEFAULT_MANAGER_PASSWORD = "admin"

    def __init__(self):
        """ Init Sessions Repository """
        self.users = dict()
        self.users_id_dict = dict()
        self._admin = UserSession(self.__load_node_manager())
        self.save_session(self._admin, self._admin.username())

    def save_session(self, user: UserMixin, key: str):
        """ Register new user session at session repository. 
            
            Args:
                user (UserMixin) : User session instance to be registered.
                key (str): Key used to identify the new user session.
        """
        self.users[key] = user
        self.users_id_dict[user.id] = self.users[user.username()]

    # Verify if already exists some user with these credentials
    def get_session(self, username: str) -> UserMixin:
        """ Retrieve user object registered.

            Args:
                username (str) : Key used to identify the desired user object.
            Returns:
                user : User object.
        """
        return self.users.get(username)

    # Recover user session by session id
    def get_session_by_id(self, sessionid: str):
        """ Retrieve user session registered at session repository.
            
            Args:
                sessionid (str) : ID of user session.
            Returns:
                session : Session instance.
        """
        return self.users_id_dict.get(sessionid)

    def authenticate(self, payload: dict):
        """ Performs authentication process.
            
            Args:
                payload (dict) : Dict structure containing user credentials.
            Returns:
                session : Returns a user session.
        """
        key = payload.get(AccountCredential.USERNAME_FIELD)
        session_object = self.users.get(key)
        if session_object:
            if session_object.authenticate(payload):
                return session_object

    def __load_node_manager(self):
        """ Start session repository with default ADMIN credentials. """
        return AccountCredential(
            SessionsRepository.DEFAULT_MANAGER_USER,
            SessionsRepository.DEFAULT_MANAGER_PASSWORD,
        )
