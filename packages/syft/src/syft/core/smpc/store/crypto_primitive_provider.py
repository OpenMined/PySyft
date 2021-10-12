"""Crypto Primitives."""

# stdlib
from collections import defaultdict
from typing import Any
from typing import Callable
from typing import DefaultDict
from typing import Dict
from typing import List


class CryptoPrimitiveProvider:
    """A trusted third party should use this class to generate crypto primitives."""

    _func_providers: Dict[str, Callable] = {}
    _ops_list: DefaultDict[str, List] = defaultdict(list)
    cache_store: Dict[Any, Any] = {}

    def __init__(self) -> None:  # noqa
        raise ValueError("This class should not be initialized")

    @staticmethod
    def generate_primitives(
        op_str: str,
        clients: List[Any],
        parties_info: List[Any],
        g_kwargs: Dict[str, Any] = {},
        p_kwargs: Dict[str, Any] = {},
    ) -> List[Any]:
        """Generate "op_str" primitives.

        Args:
            op_str (str): Operator.
            parties (List[Any]): Parties to generate primitives for.
            parties (List[Any]): Parties to generate primitives for.
            g_kwargs: Generate kwargs passed to the registered function.
            p_kwargs: Populate kwargs passed to the registered populate function.

        Returns:
            List[Any]: List of primitives.

        Raises:
            ValueError: If op_str is not registered.

        """
        if op_str not in CryptoPrimitiveProvider._func_providers:
            raise ValueError(f"{op_str} not registered")

        nr_parties = len(parties_info)

        generator = CryptoPrimitiveProvider._func_providers[op_str]
        primitives = generator(**g_kwargs, nr_parties=nr_parties)

        if p_kwargs is not None:
            """Do not transfer the primitives if there is not specified a
            values for populate kwargs."""
            CryptoPrimitiveProvider._transfer_primitives_to_parties(
                op_str=op_str,
                primitives=primitives,
                clients=clients,
                parties_info=parties_info,
                p_kwargs=p_kwargs,
            )

        # Since we do not have (YET!) the possiblity to return typed tuples from a remote
        # execute function we are using this
        return primitives

    @staticmethod
    def _transfer_primitives_to_parties(
        op_str: str,
        primitives: List[Any],
        clients: List[Any],
        parties_info: List[Any],
        p_kwargs: Dict[str, Any],
    ) -> None:
        cache_store = CryptoPrimitiveProvider.cache_store
        if not isinstance(primitives, list):
            raise ValueError("Primitives should be a List")

        if len(primitives) != len(parties_info):
            raise ValueError(
                f"Primitives length {len(primitives)} != Parties length {len(parties_info)}"
            )

        if len(clients) != len(parties_info):
            raise ValueError(
                f"Clients length {len(clients)} != Parties Information length {len(parties_info)}"
            )

        for primitives_party, client in zip(primitives, clients):
            if client not in cache_store:
                cache_store[client] = client.syft.core.smpc.store.CryptoStore()

            # TODO: This should be done better
            for primitive in primitives_party[0]:
                primitive.parties_info = parties_info

            client.syft.core.tensor.smpc.share_tensor.populate_store(
                op_str, primitives_party, **p_kwargs
            )

    @staticmethod
    def get_state() -> str:
        """Get the state of a CryptoProvider.

        Returns:
            str: CryptoProvider
        """
        res = f"Providers: {list(CryptoPrimitiveProvider._func_providers.keys())}\n"
        return res
