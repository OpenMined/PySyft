# syft.frameworks.torch.crypto package

## Submodules

## syft.frameworks.torch.crypto.securenn module

This is an implementation of the SecureNN paper
[https://eprint.iacr.org/2018/442.pdf](https://eprint.iacr.org/2018/442.pdf)

Note that there is a difference here in that our shares can be
negative numbers while they are always positive in the paper


#### syft.frameworks.torch.crypto.securenn.decompose(tensor)
decompose a tensor into its binary representation.


#### syft.frameworks.torch.crypto.securenn.flip(x, dim)
Reverse the order of the elements in a tensor


#### syft.frameworks.torch.crypto.securenn.msb(a_sh)
Compute the most significant bit in a_sh, this is an implementation of the
SecureNN paper [https://eprint.iacr.org/2018/442.pdf](https://eprint.iacr.org/2018/442.pdf)


* **Parameters**

    **a_sh** (*AdditiveSharingTensor*) – the tensor of study



* **Returns**

    the most significant bit



#### syft.frameworks.torch.crypto.securenn.private_compare(x, r, BETA)
Perform privately x > r


* **Parameters**

    * **x** (*AdditiveSharedTensor*) – the private tensor

    * **r** (*MultiPointerTensor*) – the threshold commonly held by alice and bob

    * **BETA** (*MultiPointerTensor*) – a boolean commonly held by alice and bob to
      hide the result of computation for the crypto provider



* **Returns**

    β′ = β ⊕ (x > r).



#### syft.frameworks.torch.crypto.securenn.relu(a_sh)
Compute Relu


* **Parameters**

    **a_sh** (*AdditiveSharingTensor*) – the private tensor on which the op applies



* **Returns**

    Dec(a_sh) > 0
    encrypted in an AdditiveSharingTensor



#### syft.frameworks.torch.crypto.securenn.relu_deriv(a_sh)
Compute the derivative of Relu


* **Parameters**

    **a_sh** (*AdditiveSharingTensor*) – the private tensor on which the op applies



* **Returns**

    0 if Dec(a_sh) < 0
    1 if Dec(a_sh) > 0
    encrypted in an AdditiveSharingTensor



#### syft.frameworks.torch.crypto.securenn.share_convert(a_sh)
Convert shares of a in field L to shares of a in field L - 1


* **Parameters**

    **a_sh** (*AdditiveSharingTensor*) – the additive sharing tensor who owns
    the shares in field L to convert



* **Returns**

    An additive sharing tensor with shares in field L-1


## syft.frameworks.torch.crypto.spdz module


#### syft.frameworks.torch.crypto.spdz.spdz_mul(cmd: Callable, x_sh, y_sh, crypto_provider: syft.workers.abstract.AbstractWorker, field: int)
Abstractly multiplies two tensors (mul or matmul)


* **Parameters**

    * **cmd** – a callable of the equation to be computed (mul or matmul)

    * **x_sh** (*AdditiveSharingTensor*) – the left part of the operation

    * **y_sh** (*AdditiveSharingTensor*) – the right part of the operation

    * **crypto_provider** (*AbstractWorker*) – an AbstractWorker which is used to generate triples

    * **field** (*int*) – an integer denoting the size of the field



* **Returns**

    an AdditiveSharingTensor


## Module contents
