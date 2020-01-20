from flask_login import UserMixin
import syft as sy
import uuid
from syft.grid.authentication.account import AccountCredential
from .. import hook, local_worker


class UserSession(UserMixin):

    NAMESPACE_DNS = "openmined.org"

    def __init__(self, user: AccountCredential, active=True):
        """ Handle session with User Authentication.
            
            Args:
                user (AccountCredential) : User instance.
                active (bool) : Session state.
        """
        self.id = uuid.uuid5(uuid.NAMESPACE_DNS, UserSession.NAMESPACE_DNS)
        self.user = user  # PySyft Account Credential object
        self.tensor_requests = list()

        # If it is the first session of this user at this node.
        if user.username not in hook.local_worker._known_workers:
            node_name = user.username + "_" + str(local_worker.id)
            self.node = sy.VirtualWorker(hook, id=node_name)
        else:
            self.node = hook.local_worker._known_workers[user.username]
        self.active = True

    def get_id(self):
        """ Get Session ID.
            
            Returns:
                ID: Session's ID.
        """
        return self.id

    def save_tensor_request(self, request_msg: tuple):
        """ Save tensor request at user's request list.
        
            Args:
                request_msg (tuple) : Tuple structure containing tensor id, credentials and reason.
        """
        self.tensor_requests.append(request_msg)

    @property
    def worker(self) -> sy.VirtualWorker:
        """ Get Worker used by current session.
        
            Returns:
                node (VirtualWorker) : Worker used by this session.
        """
        return self.node

    def username(self) -> str:
        """ Get username of this session.

            Returns:
                username (str) : session's username.
        """
        return self.user.username

    def is_active(self) -> bool:
        """ Get session's state.
        
            Returns:
                session_state (bool) : session's state.
        """
        return self.active

    def authenticate(self, payload: dict) -> bool:
        """ Verify if payload credentials matches with this user instance.
        
            Args:
                payload (dict) : Dict containing user credentials.
            Returns:
                result (bool) : Credential verification result.
        """
        candidate_username = payload.get(AccountCredential.USERNAME_FIELD)
        candidate_password = payload.get(AccountCredential.PASSWORD_FIELD)
        if candidate_username and candidate_password:
            return (
                self.user.password == candidate_password
                and self.user.username == candidate_username
            )
