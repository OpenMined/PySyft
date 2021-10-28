# future
from __future__ import annotations

# stdlib
import functools
import itertools
import operator
import secrets
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

# third party
import numpy as np
import numpy.typing as npt
import torch

# syft absolute
import syft as sy

# relative
from . import utils
from .... import logger
from ...smpc.protocol.spdz import spdz
from ...smpc.store import CryptoPrimitiveProvider
from ..passthrough import PassthroughTensor  # type: ignore
from ..passthrough import SupportedChainType  # type: ignore
from ..util import implements  # type: ignore
from .party import Party
from .share_tensor import ShareTensor

METHODS_FORWARD_ALL_SHARES = {
    "repeat",
    "copy",
    "diagonal",
    "flatten",
    "transpose",
    "partition",
    "resize",
    "ravel",
    "compress",
    "reshape",
    "squeeze",
    "swapaxes",
    "sum",
}
INPLACE_OPS = {
    "resize",
}

PARTIES_REGISTER_CACHE: Dict[Any, Party] = {}


class MPCTensor(PassthroughTensor):
    __slots__ = (
        "seed_przs",
        "mpc_shape",
        "parties",
        "parties_info",
        "ring_size",
    )

    def __init__(
        self,
        parties: List[Any],
        secret: Optional[Any] = None,
        shares: Optional[List[ShareTensor]] = None,
        shape: Optional[Tuple[int, ...]] = None,
        seed_przs: Optional[int] = None,
        ring_size: Optional[int] = None,
    ) -> None:

        if secret is None and shares is None:
            raise ValueError("Secret or shares should be populated!")
        if (shares is not None) and (not isinstance(shares, (tuple, list))):
            raise ValueError("Shares should be a list or tuple")

        if seed_przs is None:
            # Allow the user to specify if they want to use a specific seed when generating the shares
            # ^This is unsecure and should be used with cautioness
            seed_przs = secrets.randbits(32)

        self.seed_przs = seed_przs
        self.parties = parties
        self.parties_info = MPCTensor.get_parties_info(parties)

        if ring_size is not None:
            self.ring_size = ring_size
        else:
            self.ring_size = MPCTensor.get_ring_size_from_secret(secret, shares)
        self.mpc_shape = shape

        # TODO: We can get this from the the secret if the secret is local
        # TODO: https://app.clubhouse.io/openmined/story/1128/tech-debt-for-adp-smpc-demo?stories_sort_by\
        #  =priority&stories_group_by=WORKFLOW_STATE
        if shape is None:
            raise ValueError("Shape of the secret should be known")

        if secret is not None:
            shares = MPCTensor._get_shares_from_secret(
                secret=secret,
                parties=parties,
                parties_info=self.parties_info,
                shape=shape,
                seed_przs=seed_przs,
                ring_size=self.ring_size,
            )

        if shares is None:
            raise ValueError("Shares should not be None at this step")

        res = list(MPCTensor._mpc_from_shares(shares, parties=parties))

        # we need to make sure that when we zip up clients from
        # multiple MPC tensors that they are in corresponding order
        # so we always sort all of them by the id of the domain
        # TODO: store children as lists of dictionaries because eventually
        # it's likely that we have multiple shares from the same client
        # (For example, if you wanted a domain to have 90% share ownership
        # you'd need to produce 10 shares and give 9 of them to the same domain)
        # TODO captured: https://app.clubhouse.io/openmined/story/1128/tech-debt-for-adp-smpc-\
        #  demo?stories_sort_by=priority&stories_group_by=WORKFLOW_STATE
        res.sort(key=lambda share: share.client.name + share.client.id.no_dash)

        super().__init__(res)

    @staticmethod
    def get_ring_size_from_secret(
        secret: Optional[Any] = None, shares: Optional[List[Any]] = None
    ) -> int:
        if secret is None:
            value = shares[0]  # type: ignore
        else:
            value = secret
        if utils.ispointer(value):
            dtype = getattr(value, "public_dtype", None)
        else:
            dtype = getattr(value, "dtype", None)

        ring_size = utils.TYPE_TO_RING_SIZE.get(dtype, None)
        if ring_size is not None:
            return ring_size

        logger.warning("Ring size was not found! Defaulting to 2**32.")
        return 2 ** 32

    @staticmethod
    def get_parties_info(parties: Iterable[Any]) -> List[Party]:
        # relative
        from ....grid.client import GridHTTPConnection

        parties_info: List[Party] = []
        for party in parties:
            connection = party.routes[0].connection
            if not isinstance(connection, GridHTTPConnection):
                raise TypeError(
                    f"You tried to pass {type(connection)} for multiplication dependent operation."
                    + "Currently Syft works only with hagrid"
                    + "We apologize for the inconvenience"
                    + "We will add support for local python objects very soon."
                )
            party_info = PARTIES_REGISTER_CACHE.get(party, None)

            if party_info is None:
                base_url = connection.base_url
                url = base_url.rsplit(":", 1)[0]
                port = int(base_url.rsplit(":", 1)[1].split("/")[0])
                party_info = Party(url, port)
                PARTIES_REGISTER_CACHE[party] = party_info
                try:
                    sy.register(  # nosec
                        name="Howard Wolowtiz",
                        email="howard@mit.edu",
                        password="astronaut",
                        url=url,
                        port=port,
                        verbose=False,
                    )
                except Exception:
                    """"""
                    # TODO : should modify to return same client if registered.
                    # print("Proxy Client already User Register", e)
            parties_info.append(party_info)

        return parties_info

    def publish(self, sigma: float) -> MPCTensor:
        new_shares = []
        for share in self.child:
            new_share = share.publish(sigma=sigma)
            new_shares.append(new_share)

        return MPCTensor(
            parties=self.parties,
            shares=new_shares,
            shape=self.mpc_shape,
            seed_przs=self.seed_przs,
        )

    @property
    def shape(self) -> Optional[Tuple[int, ...]]:
        return self.mpc_shape

    @staticmethod
    def _mpc_from_shares(
        shares: List[ShareTensor],
        parties: Optional[List[Any]] = None,
    ) -> List[ShareTensor]:
        if not isinstance(shares, (list, tuple)):
            raise ValueError("_mpc_from_shares expected a list or tuple of shares")

        if utils.ispointer(shares[0]):
            # Remote shares
            return shares
        elif parties is None:
            raise ValueError(
                "Parties should not be None if shares are not already sent to parties"
            )
        else:
            return MPCTensor._mpc_from_local_shares(shares, parties)

    @staticmethod
    def _mpc_from_local_shares(
        shares: List[ShareTensor], parties: List[Any]
    ) -> List[ShareTensor]:
        # TODO: ShareTensor needs to have serde serializer/deserializer
        shares_ptr = [share.send(party) for share, party in zip(shares, parties)]
        return shares_ptr

    @staticmethod
    def _get_shares_from_secret(
        secret: Any,
        parties: List[Any],
        shape: Tuple[int, ...],
        seed_przs: int,
        parties_info: List[Party],
        ring_size: int,
    ) -> List[ShareTensor]:
        if utils.ispointer(secret):
            if shape is None:
                raise ValueError("Shape must be specified when the secret is remote")
            return MPCTensor._get_shares_from_remote_secret(
                secret=secret,
                shape=shape,
                parties=parties,
                seed_przs=seed_przs,
                parties_info=parties_info,
                ring_size=ring_size,
            )

        return MPCTensor._get_shares_from_local_secret(
            secret=secret,
            seed_przs=seed_przs,
            ring_size=ring_size,
            shape=shape,
            parties_info=parties_info,
        )

    @staticmethod
    def _get_shares_from_remote_secret(
        secret: Any,
        shape: Tuple[int, ...],
        parties: List[Any],
        seed_przs: int,
        parties_info: List[Party],
        ring_size: int,
    ) -> List[ShareTensor]:
        shares = []
        for i, party in enumerate(parties):
            if secret is not None and party == secret.client:
                value = secret
            else:
                value = None

            # relative
            from ..autodp.single_entity_phi import (
                TensorWrappedSingleEntityPhiTensorPointer,
            )

            if isinstance(secret, TensorWrappedSingleEntityPhiTensorPointer):

                share_wrapper = secret.to_local_object_without_private_data_child()
                share_wrapper_pointer = share_wrapper.send(party)

                remote_share = party.syft.core.tensor.smpc.share_tensor.ShareTensor.generate_przs_on_dp_tensor(
                    rank=i,
                    parties_info=parties_info,
                    value=value,
                    shape=shape,
                    seed_przs=seed_przs,
                    share_wrapper=share_wrapper_pointer,
                    ring_size=ring_size,
                )

            else:
                remote_share = (
                    party.syft.core.tensor.smpc.share_tensor.ShareTensor.generate_przs(
                        rank=i,
                        parties_info=parties_info,
                        value=value,
                        shape=shape,
                        seed_przs=seed_przs,
                        ring_size=ring_size,
                    )
                )

            shares.append(remote_share)

        return shares

    @staticmethod
    def _get_shares_from_local_secret(
        secret: Any,
        shape: Tuple[int, ...],
        seed_przs: int,
        parties_info: List[Party],
        ring_size: int = 2 ** 32,
    ) -> List[ShareTensor]:
        shares = []
        nr_parties = len(parties_info)
        for i in range(nr_parties):
            if i == nr_parties - 1:
                value = secret
            else:
                value = None

            local_share = ShareTensor.generate_przs(
                rank=i,
                parties_info=parties_info,
                value=value,
                shape=shape,
                seed_przs=seed_przs,
                init_clients=False,
                ring_size=ring_size,
            )

            shares.append(local_share)

        return shares

    def request(
        self,
        reason: str = "",
        block: bool = False,
        timeout_secs: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        for child in self.child:
            child.request(
                reason=reason, block=block, timeout_secs=timeout_secs, verbose=verbose
            )

    @property
    def block(self) -> "MPCTensor":
        """Block until all shares have been created."""
        for share in self.child:
            share.block

        return self

    def block_with_timeout(self, secs: int, secs_per_poll: int = 1) -> "MPCTensor":
        """Block until all shares have been created or until timeout expires."""

        for share in self.child:
            share.block_with_timeout(secs=secs, secs_per_poll=secs_per_poll)

        return self

    def reconstruct(self) -> np.ndarray:
        # TODO: It might be that the resulted shares (if we run any computation) might
        # not be available at this point. We need to have this fail well with a nice
        # description as to which node wasn't able to be reconstructued.
        # Captured: https://app.clubhouse.io/openmined/story/1128/tech-debt-for-adp-smpc-demo?\
        # stories_sort_by=priority&stories_group_by=WORKFLOW_STATE

        # for now we need to convert the values coming back to int32
        # sometimes they are floats coming from DP
        def convert_child_numpy_type(tensor: Any, np_type: type) -> Any:
            if isinstance(tensor, np.ndarray):
                return np.array(tensor, np_type)
            if hasattr(tensor, "child"):
                tensor.child = convert_child_numpy_type(
                    tensor=tensor.child, np_type=np_type
                )
            return tensor

        dtype = utils.RING_SIZE_TO_TYPE.get(self.ring_size, None)

        if dtype is None:
            raise ValueError(f"Type for ring size {self.ring_size} was not found!")

        for share in self.child:
            if not share.exists:
                raise Exception(
                    "One of the shares doesn't exist. This probably means the SMPC "
                    "computation isn't yet complete. Try again in a moment or call .block.reconstruct()"
                    "instead to block until the SMPC operation is complete which creates this variable."
                )

        local_shares = []
        for share in self.child:
            res = share.get()
            res = convert_child_numpy_type(res, dtype)
            local_shares.append(res)

        is_share_tensor = isinstance(local_shares[0], ShareTensor)

        if is_share_tensor:
            local_shares = [share.child for share in local_shares]

        result = local_shares[0]
        op = ShareTensor.get_op(self.ring_size, "add")
        for share in local_shares[1:]:
            result = op(result, share)

        if hasattr(result, "child") and isinstance(result.child, ShareTensor):
            return result.child.child

        return result

    get = reconstruct

    @staticmethod
    def hook_method(__self: MPCTensor, method_name: str) -> Callable[..., Any]:
        """Hook a framework method.

        Args:
            method_name (str): method to hook

        Returns:
            A hooked method
        """

        def method_all_shares(
            _self: MPCTensor, *args: List[Any], **kwargs: Dict[Any, Any]
        ) -> Any:

            shares = []

            for share in _self.child:
                method = getattr(share, method_name)
                new_share = method(*args, **kwargs)
                shares.append(new_share)

                dummy_res = np.empty(_self.mpc_shape)
                if method_name not in INPLACE_OPS:
                    dummy_res = getattr(np.empty(_self.mpc_shape), method_name)(
                        *args, **kwargs
                    )
                else:
                    getattr(np.empty(_self.mpc_shape), method_name)(*args, **kwargs)

                new_shape = dummy_res.shape
            res = MPCTensor(parties=_self.parties, shares=shares, shape=new_shape)
            return res

        return functools.partial(method_all_shares, __self)

    def __getattribute__(self, attr_name: str) -> Any:

        if attr_name in METHODS_FORWARD_ALL_SHARES:
            return MPCTensor.hook_method(self, attr_name)

        return object.__getattribute__(self, attr_name)

    @staticmethod
    def reshare(mpc_tensor: MPCTensor, parties: Iterable[Any]) -> MPCTensor:
        """Reshare a given secret to a superset of parties.
        Args:
            mpc_tensor(MPCTensor): input MPCTensor to reshare.
            parties(List[Any]): Input parties List.
        Returns:
            res_mpc(MPCTensor): Reshared MPCTensor.
        Raises:
            ValueError: If the input MPCTensor and input parties are same.

        Note:
        We provide an additional layer of abstraction such that,
        when computation is performed on data belonging to different parties
        The underlying secret are automatically converted into secret shares of their input.

        Assume there are two parties Parties P1,P2

        tensor_1 = data_pointer_1 (party 1 data)
        tensor_2 = data_pointer_2 (party 2 data)

        result -------> tensor_1+ tensor_1 (local computation as the data
        belongs to the same party)

        Interesting scenario is when

        result --------> tensor_1+tensor_2

        Each tensor belongs to two different parties.
        There are automatically secret shared without the user
        knowing that a MPCTensor is being created underneath.
        """
        mpc_parties = set(mpc_tensor.parties)
        parties = set(parties)
        shape = mpc_tensor.shape
        seed_przs = mpc_tensor.seed_przs
        client_map = {share.client: share for share in mpc_tensor.child}

        if mpc_parties == parties:
            raise ValueError(
                "Input parties for resharing are same as the input parties."
            )
        parties_info = MPCTensor.get_parties_info(parties)
        shares = [client_map.get(party) for party in parties]
        for i, party in enumerate(parties):
            shares[
                i
            ] = party.syft.core.tensor.smpc.share_tensor.ShareTensor.generate_przs(
                rank=i,
                parties_info=parties_info,
                value=shares[i],
                shape=shape,
                seed_przs=seed_przs,
                ring_size=mpc_tensor.ring_size,
            )

        res_mpc = MPCTensor(shares=shares, ring_size=mpc_tensor.ring_size, shape=shape, parties=parties)  # type: ignore

        return res_mpc

    @staticmethod
    def force_matching_shareholders_across_private_args(
        self: MPCTensor, other: Any
    ) -> Tuple[MPCTensor, Any]:
        """When performing SMPC across multiple private arguments, we need to ensure that all private
        variables have compatible shareholders. This entails taking each argument's list of shareholders
        (including self), computing the union of all shareholders, and then sharing/re-sharing each
        argument across all shareholders. Once all private arguments have the same set of shareholders, we
        are ready to perform private SMPC operations.

        The reason this is ok to do from a security standpoint is that, for any object, expanding the set of
        shareholders doesn't reveal any additional information about that object to new shareholders. It just gives
        new shareholders VETO power over any operations on the object, VETO power over any derivative creations from
        the object, and VETO power over the decryption of either the object or its derivatives.

        This trust model fits data science well because it means that if you're... say... adding a tensor owned by Bob
        with a tensor owned by Alice, the result should be under joint control of both Bob and Alice. That is to say,
        both Bob and Alice should have to agree in order for the result to be used or decrypted. In order to create this
        property, we must first take each argument going into said addition and expand the shareholders of both
        sides of the addition to be both Bob and Alice so that the result of the addition is also owned by Bob and
        Alice.

        For more information on this, see courses.openmined.org. https://mortendahl.github.io/ is also great.

        Args:
            self(MPCTensor): input MPCTensor to perform sanity check on.
            other (Any): input operand.
        Returns:
            Tuple[MPCTensor,Any]: Rehared Tensor values.
        """

        # If the other argument is a pointer, then it only has one "shareholder" (the domain which is storing the
        # object the pointer is pointing to). Since this is being called on an MPCTensor (self), which always has
        # multiple shareholders, then we need to share the current pointer with the shareholders of self
        # so that self and other can be used in tensor operations (such as __add__, __mul__, etc.)
        if utils.ispointer(other):

            # self parties
            parties = self.parties

            # the party of the pointer
            client = other.client

            # Since it's theoretically possible for shape information to be based on private data, we
            # track shape on pointers and MPCTensors on the client side and use client side logic to
            # determine what the resulting shape of each operation should be. That way, no shape information
            # passes from server -> client which could leak info. In this case, since we're just SMPC sharing
            # a pointer, the result has the same shape as the pointer.
            public_shape = other.public_shape

            # if the pointer doesn't have public shape something went wrong (legacy code?) so trigger an error.
            if public_shape is None:
                # TODO: Should be modified after Trask's Synthetic data PR.
                # TRASK: I'm not sure this to do is required. All pointers have public shape
                # information and so raising an exception here is appropriate.
                # I'm not confident enough to delete this to do yet though.
                raise ValueError("The input tensor pointer should have public shape.")

            # now we must calculate all of the shareholders who need to have shared ownership over 'other' before
            # we can compute on it with 'self').

            # Special Case: if the 'other' pointer is NOT pointing to any of the shareholders of 'self',
            # then we must add one additional shareholder to 'self', the owner of the pointer.
            if client not in parties:
                new_parties = [client]
                new_parties += parties
                self = MPCTensor.reshare(self, new_parties)
            else:
                new_parties = parties

            # In all cases, we must MPC share the pointer 'other' across all parties of SMPCTensor
            # Look closely and you'll see that new_parties includes the pointer's owner in all cases
            # because of the previous check. Also, even though 'other' is a Pointer, we can just pass
            # it into the MPCTensor constructor because it knows how to handle pointer arguments (it will smpc share
            # the pointer).
            other = MPCTensor(secret=other, parties=new_parties, shape=public_shape)

        # However, if the 'other' argument is instead an MPCTensor, then we must compute the union of the two sets
        # of shareholders and reshare both 'self' and 'other' so that both have the same shareholders.
        # That is to say, if self and other had 2 and 3 non-overlapping shareholders respectively, we must
        # create two new variables (overloaded to be self, and other) which both have the same 5 shareholders.
        elif isinstance(other, MPCTensor):

            p1 = set(self.parties)  # parties/shareholders in first MPCTensor
            p2 = set(other.parties)  # parties/shareholders in second MPCTensor.

            # if the two MPCTensors don't have the same parties, expand them to the union of both
            if p1 != p2:

                # calculate the union of all shareholders
                parties_union = p1.union(p2)

                # expand 'self' to include the shareholders from 'other'
                self = (
                    MPCTensor.reshare(self, parties_union)
                    if p1 != parties_union
                    else self
                )

                # expand 'other' to include the shareholders from 'self'
                other = (
                    MPCTensor.reshare(other, parties_union)
                    if p2 != parties_union
                    else other
                )

        # return a new 'self' and a new 'other' ready for MPC computation!
        # Note: if 'other' was a public value (not a Pointer or an MPCTensor), it would have been
        # unmodified in this method. It's also ready for computation but it'll end up just being
        # sent to the remote worker as a public value at a later step.
        return self, other

    @staticmethod
    def prepare_arguments_and_run_checks(
        self: MPCTensor, other: Union[int, float, np.ndarray, torch.tensor, MPCTensor]
    ) -> Any:

        # Step 1: if other is private, we need to make sure that self and other have the same shareholders.
        # if other is public, then do nothing.
        self, other = MPCTensor.force_matching_shareholders_across_private_args(
            self, other
        )

        # Step 2: calculate random seed (a single int) to send to all shareholders
        # For SMPC operations, each shareholder computes the full list of actions
        # that every shareholder will run (including what IDs those operations will have, produce,
        # etc.) They then select the subset of actions that only they (Shareholder <x>) will actually
        # execute and then executes them. In order for all parties to generate the list of actions
        # identically, they need to coordinate their random number generators (primarliy so that they generate the
        # same IDs for objects). This is the seed which provides for that coordination.
        kwargs: Dict[Any, Any] = {"seed_id_locations": secrets.randbits(64)}

        # Step 3: Check that ring size matches between self and arguments.
        # Ring size needs to match between the arguments that go in.
        # Ring size corresponds to the type of data passed in (int32 vs int64 for example). See
        # get_ring_size_from_secret for more details. More on ring sizes in courses.openmined.org
        if isinstance(other, MPCTensor):
            # if other is private then the ring size is an attribute explicitly encoded
            other_ring_size = other.ring_size

        else:
            # if other is some sort of public value (like an int or numpy array), then we need
            # to infer it from the data type using a helper function
            other_ring_size = MPCTensor.get_ring_size_from_secret(other)

        # If the ring sizes don't match then we can't do the computation so we'll trigger an error
        # before the computation occurs.
        if self.ring_size != other_ring_size:
            raise ValueError(
                f"Ring size mismatch between self {self.ring_size} and other {other_ring_size}"
            )

        # Step 4: Calculate the ring size of the result
        # TODO (from TRASK): could this actually change? Why do we need to infer this?
        ring_size = utils.get_ring_size(self.ring_size, other_ring_size)

        if not isinstance(other, MPCTensor):
            other_shares = itertools.repeat(other)
        else:
            other_shares = other.child

        # Step 5: Calculate argument shapes
        self_shape = tuple(getattr(self, "shape", (1,)))
        other_shape = tuple(getattr(other, "shape", (1,)))

        return (
            self.child,
            other_shares,
            kwargs,
            ring_size,
            self.parties,
            self.parties_info,
            self_shape,
            other_shape,
        )

    def __add__(
        self, other: Union[int, float, np.ndarray, torch.tensor, MPCTensor]
    ) -> MPCTensor:
        """Apply the "add" operation between "self" and "other".

        Args:
            y (Union[MPCTensor, torch.Tensor, float, int]): self + y

        Returns:
            MPCTensor. Result of the operation.
        """

        # Step 1: if other is private, we need to make sure that self and other have the same shareholders.
        # Step 2: calculate random seed (a single int) to send to all shareholders
        # Step 3: Check that ring size matches between self and arguments.
        # Step 4: Calculate the ring size of the result
        # Step 5: Calculate argument shapes
        (
            self_shrs,
            other_shrs,
            kwargs,
            ring_size,
            parties,
            parties_info,
            self_shape,
            other_shape,
        ) = MPCTensor.prepare_arguments_and_run_checks(self, other)

        # Step 6: Execute the SMPC operation
        # TODO: some complex hooking logic on ShareTensor means we're passing in 'a' twice
        res_shares = [a.__add__(a, b, **kwargs) for a, b in zip(self_shrs, other_shrs)]

        # Step 7: Calculate shape of result using only publicly available data (so not conditioned on private data).
        public_shape = utils.get_shape_ndarray_method_from_shapes(
            "__add__", self_shape, other_shape
        )

        # Step 8: Create the resulting MPCTensor object.
        # Note: this doesn't mean the computation has completed. This MPCTensor object is merely a pointer
        # to where the result of the MPC computation will be deposited. Call .block to wait for the computation
        # to be finished.
        result = MPCTensor(
            shares=res_shares, shape=public_shape, ring_size=ring_size, parties=parties
        )

        return result

    def __mul__(
        self, other: Union[int, float, np.ndarray, torch.tensor, MPCTensor]
    ) -> MPCTensor:

        # Step 1: if other is private, we need to make sure that self and other have the same shareholders.
        # Step 2: calculate random seed (a single int) to send to all shareholders
        # Step 3: Check that ring size matches between self and arguments.
        # Step 4: Calculate the ring size of the result
        # Step 5: Calculate argument shapes
        (
            self_shrs,
            other_shrs,
            kwargs,
            ring_size,
            parties,
            parties_info,
            self_shape,
            other_shape,
        ) = MPCTensor.prepare_arguments_and_run_checks(self, other)

        # Step 6: If other is private and we're doing private-private mul, create beaver triples.
        if isinstance(other, MPCTensor):

            CryptoPrimitiveProvider.generate_primitives(
                "beaver_mul",
                parties=parties,
                g_kwargs={
                    "a_shape": self_shape,
                    "b_shape": other_shape,
                    "parties_info": parties_info,
                },
                p_kwargs={"a_shape": self_shape, "b_shape": other_shape},
                ring_size=ring_size,
            )

        # Step 6: Execute the SMPC operation
        res_shares = [
            # TODO: some complex hooking logic on ShareTensor means we're passing in 'a' twice
            a.__mul__(a, b, self_shape, other_shape, **kwargs)
            for a, b in zip(self_shrs, other_shrs)
        ]

        # Step 7: Calculate shape of result using only publicly available data (so not conditioned on private data).
        public_shape = utils.get_shape_ndarray_method_from_shapes(
            "__mul__", self_shape, other_shape
        )

        # Step 8: Create the resulting MPCTensor object.
        # Note: this doesn't mean the computation has completed. This MPCTensor object is merely a pointer
        # to where the result of the MPC computation will be deposited. Call .block to wait for the computation
        # to be finished.
        res = MPCTensor(
            parties=self.parties,
            shares=res_shares,
            shape=public_shape,
            ring_size=self.ring_size,
        )

        return res

    def __radd__(
        self, y: Union[int, float, np.ndarray, torch.tensor, MPCTensor]
    ) -> MPCTensor:
        return self + y

    def __sub__(self, other: MPCTensor) -> MPCTensor:
        """Apply the "sub" operation between "self" and "other".

        Args:
            other (Union[MPCTensor, torch.Tensor, float, int]): self - other

        Returns:
            MPCTensor. Result of the operation.
        """

        # Step 1: if other is private, we need to make sure that self and other have the same shareholders.
        # Step 2: calculate random seed (a single int) to send to all shareholders
        # Step 3: Check that ring size matches between self and arguments.
        # Step 4: Calculate the ring size of the result
        # Step 5: Calculate argument shapes
        (
            self_shrs,
            other_shrs,
            kwargs,
            ring_size,
            parties,
            parties_info,
            self_shape,
            other_shape,
        ) = MPCTensor.prepare_arguments_and_run_checks(self, other)

        # Step 6: Execute the SMPC operation
        # TODO: some complex hooking logic on ShareTensor means we're passing in 'a' twice
        res_shares = [a.__sub__(a, b, **kwargs) for a, b in zip(self_shrs, other_shrs)]

        # Step 7: Calculate shape of result using only publicly available data (so not conditioned on private data).
        public_shape = utils.get_shape_ndarray_method_from_shapes(
            "__sub__", self_shape, other_shape
        )

        # Step 8: Create the resulting MPCTensor object.
        # Note: this doesn't mean the computation has completed. This MPCTensor object is merely a pointer
        # to where the result of the MPC computation will be deposited. Call .block to wait for the computation
        # to be finished.
        result = MPCTensor(
            shares=res_shares, shape=public_shape, ring_size=ring_size, parties=parties
        )

        return result

    def __rsub__(self, y: MPCTensor) -> MPCTensor:
        new_self = self * (-1)
        return y + new_self

    def __rmul__(
        self, y: Union[int, float, np.ndarray, torch.tensor, MPCTensor]
    ) -> MPCTensor:
        return self * y

    def __apply_private_op(
        self, other: MPCTensor, op_str: str, **kwargs: Dict[Any, Any]
    ) -> List[ShareTensor]:

        # relative
        from ..tensor import TensorPointer

        # TODO: this is hacky and won't work for all ops.
        op_method = f"__{op_str}__"

        # TODO: this is a strange place for this to live. If an unsupported op is being requested
        # we shouldn't have gotten this far.
        if op_str in {"add", "sub"}:

            # Double-check to make sure that all MPCTensor objects have the same number of shares
            # TODO: while this is in practice redundant it's also insufficient because we need to make sure
            # they actually have the same shareholders, not just the same number of shareholders.
            if len(self.child) != len(other.child):
                raise ValueError(
                    "Zipping two different lengths will drop data. "
                    + f"{len(self.child)} vs {len(other.child)}"
                )

            if not isinstance(self.child[0], TensorPointer):
                res_shares = [
                    getattr(a, op_method)(a, b, **kwargs)
                    for a, b in zip(self.child, other.child)
                ]
            # TODO: when does this get called? Why would a share not be a TensorPointer?
            else:
                op: Callable[..., Any] = getattr(operator, op_str)
                res_shares = [op(a, b) for a, b in zip(self.child, other.child)]

        else:
            raise ValueError(f"MPCTensor Private {op_str} not supported")
        return res_shares

    def __apply_public_op(
        self, y: Any, op_str: str, **kwargs: Dict[Any, Any]
    ) -> List[ShareTensor]:
        # relative
        from ..tensor import TensorPointer

        # TODO: this is hacky and won't work for all (non-duner) ops
        op_method = f"__{op_str}__"

        # TODO: this is a strange place for this allowlist to live.
        if op_str in {"mul", "matmul", "add", "sub"}:

            # TODO: it's clear that this if statement is necessary
            if not isinstance(self.child[0], TensorPointer):
                res_shares = [
                    getattr(share, op_method)(share, y, **kwargs)
                    for share in self.child
                ]
            else:
                op: Callable[..., Any] = getattr(operator, op_str)
                res_shares = [op(share, y) for share in self.child]

        else:
            raise ValueError(f"MPCTensor Public {op_str} not supported")

        return res_shares

    def __apply_op(
        self,
        y: Union[int, float, torch.Tensor, np.ndarray, MPCTensor],
        op_str: str,
    ) -> MPCTensor:
        """Apply an operation on "self" with argument "y" which can be one of many possible types.

        Args:
            y (Union[int, float, torch.Tensor, np.ndarray, MPCTensor]: tensor to apply the operation.
            op_str (str): the operation.

        Returns:
            MPCTensor. the operation "op_str" applied on "self" and "y"
        """

        # Step 1: if y is private, we need to make sure that x and y have the same shareholders.
        # if y is public, then do nothing.
        x, y = MPCTensor.force_matching_shareholders_across_private_args(self, y)

        # Step 2: For SMPC operations, each shareholder computes the full list of actions
        # that every shareholder will run (including what IDs those operations will have, produce,
        # etc.) They then select the subset of actions that only they (Shareholder <x>) will actually
        # execute and then executes them. In order for all parties to generate the list of actions
        # identically, they need to coordinate their random number generators (primarliy so that they generate the
        # same IDs for objects). This is the seed which provides for that coordination.
        kwargs: Dict[Any, Any] = {"seed_id_locations": secrets.randbits(64)}

        # Step 3: Check that ring size matches between self and arguments.
        # Ring size needs to match between the arguments that go in.
        # Ring size corresponds to the type of data passed in (int32 vs int64 for example). See
        # get_ring_size_from_secret for more details. More on ring sizes in courses.openmined.org
        if isinstance(y, MPCTensor):
            # if y is private then the ring size is an attribute explicitly encoded
            y_ring_size = y.ring_size

        else:
            # if y is some sort of public value (like an int or numpy array), then we need
            # to infer it from the data type using a helper function
            y_ring_size = MPCTensor.get_ring_size_from_secret(y)

        # If the ring sizes don't match then we can't do the computation so we'll trigger an error
        # before the computation occurs.
        if self.ring_size != y_ring_size:
            raise ValueError(
                f"Ring size mismatch between self {self.ring_size} and other {y_ring_size}"
            )

        # Step 4: Execute computation

        # If y is private, then run the private op algorithm
        if isinstance(y, MPCTensor):
            resulting_shares = x.__apply_private_op(y, op_str, **kwargs)
        # If y is public tensor, then run the simpler public op algorithm
        else:
            resulting_shares = x.__apply_public_op(y, op_str, **kwargs)

        # Step 5: Calculate shape of result using only publicly available data (so not conditioned on private data).

        # For this step, first we need to get shape information from y. Usually this is an attribute 'shape'
        # but in the case that it isn't available, we'll default to (1,) such as if a scalar is passed in. Honestly
        # this is an opportunity for bugs and needs to be addressed. TODO: more concrete support for inferring shape.
        y_shape = getattr(y, "shape", (1,))

        # infer the shape of the result using only public information about the shape of the inputs
        # to the operation and the operation itself.
        shape = utils.get_shape(op_str, self.shape, y_shape)

        # Step 6: Calculate the ring size of the result
        # TODO (from TRASK): could this actually change? Why do we need to infer this?
        ring_size = utils.get_ring_size(self.ring_size, y_ring_size)

        # Step 7: Create the resulting MPCTensor object.
        # Note: this doesn't mean the computation has completed. This MPCTensor object is merely a pointer
        # to where the result of the MPC computation will be deposited. Call .block to wait for the computation
        # to be finished.
        result = MPCTensor(
            shares=resulting_shares, shape=shape, ring_size=ring_size, parties=x.parties
        )

        return result

    def __gt__(
        self, y: Union[int, float, np.ndarray, torch.tensor, MPCTensor]
    ) -> MPCTensor:

        self, y = MPCTensor.force_matching_shareholders_across_private_args(self, y)

        res_shares = spdz.gt_master(self, y, "mul")
        y_shape = getattr(y, "shape", (1,))
        new_shape = utils.get_shape("gt", self.mpc_shape, y_shape)
        res = MPCTensor(parties=self.parties, shares=res_shares, shape=new_shape)

        return res

    def __matmul__(
        self, y: Union[int, float, np.ndarray, torch.tensor, "MPCTensor"]
    ) -> MPCTensor:
        """Apply the "matmul" operation between "self" and "y"
        Args:
            y (Union[int, float, np.ndarray, torch.tensor, "MPCTensor"]): self @ y
        Returns:
            MPCTensor: Result of the opeartion.
        """
        if isinstance(y, ShareTensor):
            raise ValueError("Private matmul not supported yet")

        res = self.__apply_op(y, "matmul")
        return res

    def __str__(self) -> str:
        res = "MPCTensor"
        for share in self.child:
            res = f"{res}\n\t{share}"

        return res

    def __repr__(self) -> str:
        out = "MPCTensor"
        out += ".shape=" + str(self.shape) + "\n"
        for i, child in enumerate(self.child):
            out += f"\t .child[{i}] = " + child.__repr__() + "\n"
        out = out[:-1] + ""

        return out

    def put(
        self,
        indices: npt.ArrayLike,
        values: npt.ArrayLike,
        mode: Optional[str] = "raise",
    ) -> MPCTensor:
        """Performs Numpy put operation on the underlying ShareTensors.
        Args:
            indices (npt.ArrayLike): Target indices, interpreted as integers.
            values (npt.ArrayLike): Values to place at target indices.
            mode (Optional[str]): Specifies how out-of-bounds indices will behave.
        Returns:
            res (MPCTensor): Result of the operation.
        """
        shares = []
        shares.append(self.child[0].put(indices, values, mode))
        # since the value is public we assign directly to prevent overhead of random share creation.
        zero = np.zeros_like(values)
        for share in self.child[1::]:
            shares.append(share.put(indices, zero.copy(), mode))

        res = MPCTensor(shares=shares, parties=self.parties, shape=self.shape)
        return res


@implements(MPCTensor, np.add)
def add(x: np.ndarray, y: MPCTensor) -> SupportedChainType:
    return y.add(x)


@implements(MPCTensor, np.subtract)
def sub(x: np.ndarray, y: MPCTensor) -> SupportedChainType:
    return y.rsub(x)


@implements(MPCTensor, np.multiply)
def mul(x: np.ndarray, y: MPCTensor) -> SupportedChainType:
    return y.mul(x)


# @implements(MPCTensor, np.greater)
# def mul(x: np.ndarray, y: MPCTensor) -> SupportedChainType:
#     return y.gt(x)
