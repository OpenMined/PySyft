import random # to generate ids
import warnings
import phe as paillier
import torch
from phe.util import powmod

class PaillierEncryption():
    """ A PaillierEncryption class, that holds the private and public keys for paillier
    encryption. It also holds the id of tensors encrypted using this public keys generated
    by this class.

    Parameters
    ----------
    length : int
        Length of the key to be generated for the private and public key. We recommend
        a length of 22-24, due to the overflowing issues with LongTensors in PyTorch.
    owner : sy.VirtualWorker
        Owner of the private key, on default it is set to local.
    private_key_string : phe.paillier.PaillierPrivateKey
        If you already have a Paillier Private key, you can give it as a parameter,
        and can generate a new public key corresponding to that private key.

    Attributes
    ----------
    public_key : phe.paillier.PaillierPublicKey
        Public key used for encrypting data.
    private_key : phe.paillier.PaillierPrivateKey
        Private key used to decrypt data.
    tensors_encrypted : list
        An array of tensors encrypted using this public key.
    id : int
        ID of the pair of keys
    owner : sy.VirtualWorker
        Owner of the private key.

    """


    def __init__(self, length = 30, owner=None, private_key_string=None):
        ## if length of 30, the largest number could be stored on the tensors is 99999999
        warnings.warn("\nPossible data losses may occur if numbers greater 1e8 are stored with key length of 30 are encrypted during decryption due to how torch stores long tensors.", UserWarning)
        if not private_key_string == None: ## if private_key string is provided then create a private and public key using the private key string
            self.public_key, self.private_key = paillier.generate_paillier_keypair(private_key_string, length)
        else:
            ## else create a public key and private key according to the n_length
            self.public_key, self.private_key = paillier.generate_paillier_keypair(n_length=length)
        self.id = int(10e10 * random.random()) # generate an id for the encryption set
        self.owner = owner if owner != None else 'local' # set the owner for the keys
        self.tensors_encrypted = [] # list of tensors encrypted using this public key

    def __repr__(self):
        """Representation of the KeyPair.

        Returns
        -------
        rep : string
            Description of the object

        """
        rep = f"<Paillier Encryption with ID of {self.id} and owner of {self.owner}>"
        return rep


    def _tensors(self):
        """Return a list of tensors that are encrypted using this object

        Returns
        -------
        res : string
            A list of tensors that have used the public key from this object for
            encryption

        """
        if len(self.tensors_encrypted) == 0: # if no tensors exists in the list, return none
            return None
        tensors_str = ""
        for tensor in self.tensors_encrypted: # add tensors' ids to the list for returning
            if tensor == self.tensors_encrypted[-1]:
                tensors_str += f"Tensor id: {tensor}"
            else:
                tensors_str += f"Tensor id: {tensor},\n"
        if len(self.tensors_encrypted) == 1:
            res = "Tensor encrypted using this public key is: ["+tensors_str+"]"
        else:
            res = "Tensors encrypted using this public key are: ["+tensors_str+"]"
        return res

    @staticmethod
    def getTensorID(tensor):
        """Given a tensor it returns its ID which is unique, and represents its
        memory location

        Parameters
        ----------
        tensor : torch.Tensor
            PyTorch Tensor which we want to get its ID

        Returns
        -------
        tensor_id : int
            A unique ID for the tensor, which stays the same until it is deleted

        """
        tensor_id = tensor.data_ptr() # each tensor has a pointer, which is unique.
                                      # using this we can keep track of which tensors
                                      # are encrypted using the public key or not
        return tensor_id

    def getEncryptedTensors(self):
        """ Returns a list of encrypted tensors that use this public key. This
        is used for PaillierTensors to check if they share the same public key or
        not, and also to be added to the list if they use this public key.

        Returns
        -------
        type
            Description of returned object.

        """
        return self.tensors_encrypted
