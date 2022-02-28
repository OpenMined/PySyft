"""Beaver Triples Protocol.

D. Beaver. *Efficient multiparty protocols using circuit randomization*.
In J. Feigenbaum, editor, CRYPTO, volume **576** of Lecture Notes in
Computer Science, pages 420–432. Springer, 1991.
"""


# stdlib
from copy import deepcopy
import secrets
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

# third party
import numpy as np

# relative
from ....tensor.smpc.mpc_tensor import MPCTensor
from ....tensor.smpc.share_tensor import ShareTensor
from ....tensor.smpc.utils import RING_SIZE_TO_TYPE
from ...store import register_primitive_generator
from ...store import register_primitive_store_add
from ...store import register_primitive_store_get
from ...store.exceptions import EmptyPrimitiveStore

ttp_generator = np.random.default_rng()


def _get_triples(
    op_str: str,
    nr_parties: int,
    parties_info: List[Any],
    a_shape: Tuple[int],
    b_shape: Tuple[int],
    nr_instances: int = 1,
    ring_size: int = 2**32,
    **kwargs: Dict[Any, Any],
) -> Tuple[Tuple[Tuple[ShareTensor, ShareTensor, ShareTensor]]]:
    """Get triples.

    The Trusted Third Party (TTP) or Crypto Provider should provide this triples Currently,
    the one that orchestrates the communication provides those triples.".

    Args:
        op_str (str): Operator string.
        nr_parties (int): Number of parties
        parties_info (List[Any]): Parties connection information.
        a_shape (Tuple[int]): Shape of a from beaver triples protocol.
        b_shape (Tuple[int]): Shape of b part from beaver triples protocol.
        ring_size (int) : Ring Size of the triples to generate.
        kwargs: Arbitrary keyword arguments for commands.

    Returns:
        List[List[3 x List[ShareTensor, ShareTensor, ShareTensor]]]:
        The generated triples a,b,c for each party.

    Raises:
        ValueError: If the triples are not consistent.
        ValueError: If the share class is invalid.
    """
    # relative
    from ..... import Tensor

    numpy_type = RING_SIZE_TO_TYPE[ring_size]
    cmd = ShareTensor.get_op(ring_size, op_str)

    min_value, max_value = ShareTensor.compute_min_max_from_ring(ring_size)

    triples = []
    for _ in range(nr_instances):
        seed_przs = secrets.randbits(32)
        a_rand = Tensor(
            ttp_generator.integers(
                low=min_value,
                high=max_value,
                size=a_shape,
                endpoint=True,
                dtype=numpy_type,
            )
        )

        a_shares = MPCTensor._get_shares_from_local_secret(
            secret=deepcopy(a_rand),
            parties_info=parties_info,  # type: ignore
            shape=a_shape,
            seed_przs=seed_przs,
            ring_size=ring_size,
        )
        seed_przs = secrets.randbits(32)
        b_rand = Tensor(
            ttp_generator.integers(
                low=min_value,
                high=max_value,
                size=b_shape,
                endpoint=True,
                dtype=numpy_type,
            )
        )

        b_shares = MPCTensor._get_shares_from_local_secret(
            secret=deepcopy(b_rand),
            parties_info=parties_info,  # type: ignore
            shape=b_shape,
            seed_przs=seed_przs,
            ring_size=ring_size,
        )
        seed_przs = secrets.randbits(32)
        # TODO: bitwise and on passthorough tensor raises
        # hence we do it on numpy array itself.
        c_val = Tensor(cmd(a_rand.child, b_rand.child))
        c_shares = MPCTensor._get_shares_from_local_secret(
            secret=deepcopy(c_val),
            parties_info=parties_info,  # type: ignore
            shape=c_val.shape,  # type: ignore
            seed_przs=seed_przs,
            ring_size=ring_size,
        )

        # We are always creating an instance
        triples.append((a_shares, b_shares, c_shares))

    """
    Example -- for n_instances=2 and n_parties=2:
    For Beaver Triples the "res" would look like:
    res = [
        ([a0_sh_p0, a0_sh_p1], [b0_sh_p0, b0_sh_p1], [c0_sh_p0, c0_sh_p1]),
        ([a1_sh_p0, a1_sh_p1], [b1_sh_p0, b1_sh_p1], [c1_sh_p0, c1_sh_p1])
    ]

    We want to send to each party the values they should hold:
    primitives = [
        [[a0_sh_p0, b0_sh_p0, c0_sh_p0], [a1_sh_p0, b1_sh_p0, c1_sh_p0]], # (Row 0)
        [[a0_sh_p1, b0_sh_p1, c0_sh_p1], [a1_sh_p1, b1_sh_p1, c1_sh_p1]]  # (Row 1)
    ]

    The first party (party 0) receives Row 0 and the second party (party 1) receives Row 1
    """

    res_triples = list(zip(*[zip(*triple) for triple in triples]))

    return res_triples  # type: ignore


# Beaver Operations defined for Multiplication


@register_primitive_generator("beaver_mul")
def get_triples_mul(
    *args: List[Any], **kwargs: Dict[Any, Any]
) -> Tuple[Tuple[Tuple[ShareTensor, ShareTensor, ShareTensor]]]:
    """Get the beaver triples for the multiplication operation.

    Args:
        *args (List[ShareTensor]): Named arguments of :func:`beaver.__get_triples`.
        **kwargs (List[ShareTensor]): Keyword arguments of :func:`beaver.__get_triples`.

    Returns:
        Tuple[Tuple[ShareTensor, ShareTensor, ShareTensor]]: The generated triples a,b,c
        for the mul operation.
    """
    return _get_triples("mul", *args, **kwargs)  # type: ignore


@register_primitive_store_add("beaver_mul")
def mul_store_add(
    store: Dict[str, List[Any]],
    primitives: List[Any],
    a_shape: Tuple[int],
    b_shape: Tuple[int],
    ring_size: int,
) -> None:
    """Add the primitives required for the "mul" operation to the CryptoStore.

    Arguments:
        store (Dict[str, List[Any]]): the CryptoStore
        primitives (List[Any]): the list of primitives
        a_shape (Tuple[int]): the shape of the first operand
        b_shape (Tuple[int]): the shape of the second operand
    """
    config_key = f"beaver_mul_{a_shape}_{b_shape}_{ring_size}"
    if config_key in store:
        store[config_key].extend(list(primitives))
    else:
        store[config_key] = list(primitives)


@register_primitive_store_get("beaver_mul")
def mul_store_get(
    store: Dict[str, List[Any]],
    a_shape: Tuple[int, ...],
    b_shape: Tuple[int, ...],
    ring_size: int,
    remove: bool = True,
) -> Any:
    """Retrieve the primitives from the CryptoStore.

    Those are needed for executing the "mul" operation.

    Args:
        store (Dict[str, List[Any]]): The CryptoStore.
        a_shape (Tuple[int]): The shape of the first operand.
        b_shape (Tuple[int]): The shape of the second operand.
        remove (bool): True if the primitives should be removed from the store.

    Returns:
        Any: The primitives required for the "mul" operation.

    Raises:
        EmptyPrimitiveStore: If no primitive in the store for config_key.
    """
    config_key = f"beaver_mul_{tuple(a_shape)}_{tuple(b_shape)}_{ring_size}"

    try:
        primitives = store[config_key]
    except KeyError:
        raise EmptyPrimitiveStore(f"{config_key} does not exists in the store")

    try:
        primitive = primitives[0]
    except Exception:
        raise EmptyPrimitiveStore(f"No primitive in the store for {config_key}")

    if remove:
        del primitives[0]

    return primitive


# Beaver Operations defined for Matrix Multiplication


@register_primitive_generator("beaver_matmul")
def get_triples_matmul(
    *args: List[Any], **kwargs: Dict[Any, Any]
) -> Tuple[List[ShareTensor], List[ShareTensor], List[ShareTensor]]:
    """Get the beaver triples for the matmul  operation.

    Args:
        *args (List[ShareTensor]): Named arguments of :func:`beaver.__get_triples`.
        **kwargs (List[ShareTensor]): Keyword arguments of :func:`beaver.__get_triples`.

    Returns:
        Tuple[Tuple[ShareTensor, ShareTensor, ShareTensor]]: The generated triples a,b,c
        for the matmul operation.
    """
    return _get_triples("matmul", *args, **kwargs)  # type: ignore


@register_primitive_store_add("beaver_matmul")
def matmul_store_add(
    store: Dict[str, List[Any]],
    primitives: List[Any],
    a_shape: Tuple[int],
    b_shape: Tuple[int],
    ring_size: int,
) -> None:
    """Add the primitives required for the "matmul" operation to the CryptoStore.

    Args:
        store (Dict[str, List[Any]]): The CryptoStore.
        primitives (List[Any]): The list of primitives
        a_shape (Tuple[int]): The shape of the first operand.
        b_shape (Tuple[int]): The shape of the second operand.

    """
    config_key = f"beaver_matmul_{a_shape}_{b_shape}_{ring_size}"
    if config_key in store:
        store[config_key].extend(list(primitives))
    else:
        store[config_key] = list(primitives)


@register_primitive_store_get("beaver_matmul")
def matmul_store_get(
    store: Dict[str, List[Any]],
    a_shape: Tuple[int, ...],
    b_shape: Tuple[int, ...],
    ring_size: int,
    remove: bool = True,
) -> Any:
    """Retrieve the primitives from the CryptoStore.

    Those are needed for executing the "matmul" operation.

    Args:
        store (Dict[str, List[Any]]): The CryptoStore.
        a_shape (Tuple[int]): The shape of the first operand.
        b_shape (Tuple[int]): The shape of the second operand.
        remove (bool): True if the primitives should be removed from the store.

    Returns:
        Any: The primitives required for the "matmul" operation.

    Raises:
        EmptyPrimitiveStore: If no primitive in the store for config_key.
    """
    config_key = f"beaver_matmul_{tuple(a_shape)}_{tuple(b_shape)}_{ring_size}"

    try:
        primitives = store[config_key]
    except KeyError:
        raise EmptyPrimitiveStore(f"{config_key} does not exists in the store")

    try:
        primitive = primitives[0]
    except Exception:
        raise EmptyPrimitiveStore(f"No primitive in the store for {config_key}")

    if remove:
        del primitives[0]

    return primitive
