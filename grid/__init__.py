import syft

from grid.websocket_client import WebsocketGridClient
from grid import utils as gr_utils
from grid import deploy
from grid.grid_network import GridNetwork

from grid.utils import connect_all_nodes
from grid.auth import auth_credentials as credentials
from grid import auth
import warnings

__all__ = ["workers", "connect_all_nodes", "syft"]

warnings.warn(
    "This library is DEPRECATED and should be deleted soon. To use the grid features use the syft.grid module.",
    Warning,
)

# ======= Providing a friendly API on top of Syft ===============
def encrypt(self, worker_1, worker_2, crypto_provider):
    """tensor.fix_prec().share()"""
    return self.fix_prec().share(worker_1, worker_2, crypto_provider=crypto_provider)


syft.frameworks.torch.tensors.interpreters.native.TorchTensor.encrypt = encrypt
syft.messaging.plan.Plan.encrypt = encrypt


def request_decryption(self):
    """tensor.get().float_prec()"""
    return self.get().float_prec()


syft.frameworks.torch.tensors.interpreters.native.TorchTensor.request_decryption = (
    request_decryption
)


# ========= Set up User credentials ============================


def load_credentials(directory=None, folder=None):
    """ Load and parse files to set grid credentials.

        Args:
            directory (str) : file's path (DEFAULT: /home/<user>).
            folder (str) : folder name (DEFAULT: .openmined).
    """
    auth.config.read_authentication_configs(directory, folder)
