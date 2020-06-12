from pyfe.encryptor import Encryptor

from syft.generic.abstract.tensor import AbstractTensor


class FunctionalEncryptedTensor(AbstractTensor):
    def __init__(self, owner=None, id=None, tags=None, description=None):
        """Initialize a FunctionalEncryptedTensor, whose behaviour is to encrypt a
        tensor, on which functional encryption can be used

        Args:
            owner: An optional BaseWorker object to specify the worker on which
                the tensor is located.
            id: An optional string or integer id of the FunctionalEncryptedTensor.
            tags: an optional set of hashtags corresponding to this tensor
                which this tensor should be searchable for
            description: an optional string describing the purpose of the
                tensor
         """
        super().__init__(id=id, owner=owner, tags=tags, description=description)
        self.scheme = None

    def encrypt(self, context, public_key):
        """
        This method will encrypt each value in the tensor using Functional encryption.

        Args:
            context: a PyFE Context created using
                pyfe.context.Context()
            public_key: a PublicKey object
        """
        output = FunctionalEncryptedTensor()
        output.child = self.child
        output.encrypt_(context, public_key)

    def encrypt_(self, context, public_key):
        """
            This method will encrypt each value in the tensor using Functional encryption.

        Args:
            context: a PyFE Context created using
                pyfe.context.Context()
            public_key: a PublicKey object
        """

        input = self.child.flatten().tolist()

        # We are required to provide pk manually by user , key generation requires length of data
        # TODO: Find solution for this

        encryptor = Encryptor(context, public_key)

        new_child = encryptor.encrypt(input)

        self.child = new_child
