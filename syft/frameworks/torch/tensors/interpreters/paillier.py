from syft.frameworks.torch.utils.paillier import PaillierEncryption
from syft.frameworks.torch.tensors.interpreters.abstract import AbstractTensor
from syft.frameworks.torch.overload_torch import overloaded
import warnings
import phe as paillier
from phe.util import powmod
import torch as th


class PaillierTensor(AbstractTensor):
    def __init__(self, worker = None, is_encrypted = False,is_prec=False, prec=0, owner=None, id=None, tags=None, description=None):
        """Initializes a Paillier Encrypted Tensor. This tensor can be sent to a remote
        worker for computation and received for calculations while keeping the data
        encrypted. Paillier Encryption does not support decimal and negative numbers
        but few tricks can be used to provide encoding for such numbers. For decimal
        numbers we can use fix precision tensors, so we convert them to integers, and
        for negative numbers they are encoded in such a way that when they are encoded
        they are greater than the max_int, so we can treat them as overflowed numbers
        and by subtracting max_int we can properly restore them. For the implementation
        of decryption of negative numbers please refer to _neg_decrypt method and for
        decimal numbers refer to _to_fix_prec method.

        Args:
            public_key: This is the public that will be used for encrypting the tensor
        """
        super().__init__(id=id, owner=owner, tags=tags, description=description)
        self.worker = worker
        self.is_prec = is_prec # if it is fixed precision type
        self.prec = prec # tensor's precision
        self.is_encrypted = is_encrypted # whether the tensor is encrypted or not




    def _encrypt(self, num):
        """ Given a number it encrypts it using the public key provided. Since,
        currently the largest dtype provided by th is the LongTensor (int64),
        it also checks for overflow.

        # NOTE: Large Precision Tensor is going to be created in the near future,
        which can solve this problem.
        https://github.com/OpenMined/PySyft/pull/2147

        Args:
            num (int): Number to be encrypted.

        Returns:
            encrypted_num (int): Encrypted number.

        """

        encrypted_num = self.public_key.raw_encrypt(num)
        if encrypted_num >= 2**64:
            raise OverflowError("An overflow has occurred, try to make the length of the encryption shorter")
        return encrypted_num



    def encryptTensor(self):
        """Given a tensor, it performs paillier encryption on it. This method,
        also adds the current tensor to the list of tensors encrypted by this
        public key.

        Args:
            encrypted_tensors (int): Description of parameter `encrypted_tensors`.
        """
        # encrypts a tensor using the public key
        if self.child.owner._has_paillier_public_key:
            self.public_key = self.child.owner._paillier_public_key
        else:
            self.child.owner._generate_paillier_encryption()
            self.public_key = self.child.owner._paillier_public_key

        self.child.owner._encrypted_tensors.append(self.getTensorID(self.child)) # firstly we add the tensor's id
                                                           # to the list of encrypted tensors

        for i in range(len(self.child.view(-1))): # we loop over all elements, and encrypt them
                                                  # using the public key
            self.child.view(-1)[i] = self._encrypt(int(self.child.view(-1)[i]))
        self.encrypted = True
        return self

    def _neg_decrypt(self, num):
        """Since Paillier cannot encrypt negative numbers, the paillier module used
        a very clever trick which doubles the encryption field and all negative numbers
        are stored in the overflowing field (aka if a number is negative, its decrypted
        value is the limit of the encryption or public_key.n + value of the number)

        Args:
            num (int): Private key to be used for decryption of provided numbers.


        Returns:
            int: Decrypted number.

        """
        if self.child.owner._paillier_private_key.raw_decrypt(num) > self.child.owner._paillier_public_key.max_int: # check to see if the
                                                                   # decrypted number is
                                                                   # negative or not
            return self.child.owner._paillier_private_key.raw_decrypt(num) - self.child.owner._paillier_public_key.n # if it is then return
                                                                    # its actual value
        else:
            return self.child.owner._paillier_private_key.raw_decrypt(num) # otherwise decrypt it using the private key

    @staticmethod
    def getTensorID(tensor):
        """Given a tensor it returns its ID which is unique, and represents its
        memory location

        Args:
            tensor (th.Tensor): Python Tensor which we want to get its ID

        Returns:
            tensor_id (int): A unique ID for the tensor, which stays the same until it is deleted

        """
        if hasattr(tensor, 'child'):
            tensor.(tensor.child) 
        else:
            tensor_id = tensor.data_ptr() # each tensor has a pointer, which is unique.
                                      # using this we can keep track of which tensors
                                      # are encrypted using the public key or not
            return tensor_id

    def _scalar_add(self, encrypted_num, other_num): # performs addition of paillier encrypted
                                                     # numbers
        return encrypted_num * other_num % self._paillier_public_key.nsquare

    # @overloaded.methods cannot be overloaded because of self.child.owner.encrypted_tensors is a parameter
    def add(self, other):
        if hasattr(other, 'child'):
            if not isinstance(other.child, PaillierTensor): # check to see if tensor is
                                                            # encrypted, if not encrypt
                                                            # it with the same public key
                        other = PaillierTensor().on(other)

            elif self.getTensorID(other.child.child) not in self.child.owner.encrypted_tensors: # check to see if the tensor shares
                                                                           # the same encryption key with self
                raise TypeError("Tensor is not encrypted with the same public_key")

            elif isinstance(other.child, PaillierTensor):
                res = PaillierTensor().on(th.zeros(other.child.shape, dtype = th.int64))
                self.child.owner.encrypted_tensors.append(self.getTensorID(res.child.child)) # add the result tensor to the list of
                                                                                             # encrypted tensors
            for i in range(len(self.child.view(-1))): # perform the addition
                res.child.child.view(-1)[i] = self._scalar_add(int(self.child.view(-1)[i]), int(other.child.child.view(-1)[i]))
            return res
        else:
            other = PaillierTensor().on(other)
            other.child.encryptTensor(self.child.owner.encrypted_tensors)
            
        res = PaillierTensor().on(th.zeros(other.child.shape, dtype=th.int64)) # create a new tensor for the result

        self.child.owner.encrypted_tensors.append(self.getTensorID(res.child.child)) # add the result tensor to the list of
                                                                # encrypted tensors
        for i in range(len(self.child.view(-1))): # perform the addition
            res.child.child.view(-1)[i] = self._scalar_add(int(self.child.view(-1)[i]), int(other.child.child.view(-1)[i]))
        return res

    def sub(self, other): # performs matrix subtraction
        if hasattr(other, 'child'):
            if isinstance(other.child, PaillierTensor):
                other = other.child.mul(-1)
        else:
            other = other * -1
        return self.add(other)


    def _new_mult(self, encrypted_num, num): # performs multiplication of paillier encrypted
                                             # numbers
        return powmod(encrypted_num, num, self._paillier_public_key.nsquare)


    def mul(self, other):
        other = self.fix_prec(other, 'mult')
        res = th.zeros(self.child.shape, dtype=th.int64)
        self.child.owner.encrypted_tensors.append(self.getTensorID(res))
        for i in range(len(self.child.view(-1))):
            res.view(-1)[i] = self._new_mult(int(self.child.view(-1)[i]), other)
        res = PaillierTensor(public_key = self._paillier_public_key, is_prec = self.is_prec, prec = self.prec).on(res)
        return res

    def div(self, other): # performs scalar division of matrices
        if isinstance(other, th.Tensor):
            raise NotImplemented("Only scalar operations are supported.")
        other = int(1./other)
        if not isinstance(other, int):
            other = self.fix_prec(other, 'div')
        res = self.mul(other)
        return res


    def fix_prec(self, other, op = 'add'): # if a scalar is needed to be a fixed precision, this function
                                           # turns it into a fixed precision type
        if self.is_prec == True and self.prec > 12:
            raise OverflowError("Overflow has happened due to fix precision encoding.")
        prec = len(str(other)[str(other).find('.')+1:])
        prec = 3 if prec >= 3 else prec
        if self.is_prec == True: # if operation is multiplication the
                                 # precision should be added to the
                                 # existing precision, if it is
                                 # division it should be subtracted
                                 # from the existing one. if it's
                                 # addition or subtraction no action
                                 # is to needed to be taken.
            if op == 'mult':
                self.prec +=  prec
                warnings.warn("May cause overflow, make sure you are not working with float/double numbers since paillier cryptosystem does not support them.")
            if op == 'div':
                self.prec -= prec
        self.is_prec = True
        self.prec = prec
        return int(other*pow(10,self.prec))


    def decryptTensor(self): # decrypts tensors using the private key
        # encrypted_tensors.remove(self.getTensorID(self.child)) 
        # # if tensor is in the encrypted list, then remove it
        if not self.child.owner._has_paillier_public_key:
            raise AttributeError("The owner of this tensor does not have the private key")
        for i in range(len(self.child.view(-1))):
            self.child.view(-1)[i] = self._neg_decrypt(int(self.child.view(-1)[i]))
        if self.is_prec:
            self.child = self.child/10**self.prec
            self.prec = 0
            self.is_prec = False
        self.encrypted = False
        return self

