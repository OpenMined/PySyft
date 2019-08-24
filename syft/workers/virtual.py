from .base import BaseWorker
from ..frameworks.torch.utils import PaillierEncryption


class VirtualWorker(BaseWorker):
    def _send_msg(self, message: bin, location: BaseWorker) -> bin:
        return location._recv_msg(message)

    def _recv_msg(self, message: bin) -> bin:
        return self.recv_msg(message)

    def _generate_paillier_encryption(self, length = 14):
        self.encryption = PaillierEncryption(length=length)
        self._has_paillier_private_key = True
        self._has_paillier_public_key = True
        self._paillier_public_key = self.encryption.public_key
        self._paillier_private_key = self.encryption.private_key
        self._encrypted_tensors = []
    
    def _set_paillier_public_key(self, public_key):
        self._paillier_public_key = public_key
        self._has_paillier_public_key = True

    def _set_paillier_private_key(self, private_key):
        self._paillier_private_key = private_key
        self._has_paillier_private_key = True
