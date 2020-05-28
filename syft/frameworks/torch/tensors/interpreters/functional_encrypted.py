from syft.generic.tensor import AbstractTensor
import pyfe as fe


class FunctionalEncryptedTensor(AbstractTensor):
    def __init__(self, owner=None, id=None, tags=None, description=None):
        """Initialize a FunctionalEncryptedTensor, whose behaviour is to encrypt a
        tensor, on which functional encryption can be used

        Args:
            owner: An optional BaseWorker object to specify the worker on which
                the tensor is located.
            id: An optional string or integer id of the CKKSTensor.
            tags: an optional set of hashtags corresponding to this tensor
                which this tensor should be searchable for
            description: an optional string describing the purpose of the
                tensor
         """
        super().__init__(id=id, owner=owner, tags=tags, description=description)
        self.scheme = None

    def encrypt(self, scheme):
        """
        This method will encrypt each value in the tensor using Functional encryption.

        Args:
            scheme: a PyFE Scheme created using
                pyfe.scheme.Scheme()
        """
        output = FunctionalEncryptedTensor()
        output.child = self.child
        output.encrypt_(scheme)

    def encrypt_(self, scheme):
        """
            This method will encrypt each value in the tensor using Functional encryption.

        Args:
            scheme: a PyFE Scheme created using
                pyfe.scheme.Scheme()
        """
        input = self.child.flatten().tolist()
        pk, msk = scheme.setup(vector_length=len(input))

        new_child = scheme.encrypt(pk, fe.vectors.Vector(input))
        self.child = new_child
        self.scheme = scheme
