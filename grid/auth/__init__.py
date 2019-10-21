import os.path
from .user_auth import UserAuthentication

BASE_DIR = os.path.expanduser("~")
BASE_FOLDER = ".openmined"
AUTH_MODELS = [UserAuthentication]
auth_credentials = []

from .config import read_authentication_configs, search_credential
