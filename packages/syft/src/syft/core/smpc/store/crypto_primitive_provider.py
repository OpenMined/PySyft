"""Crypto Primitives."""

# stdlib
from collections import defaultdict
from typing import Any
from typing import Callable
from typing import DefaultDict
from typing import Dict
from typing import List
from typing import Optional

# relative
from ...common import UID


class CryptoPrimitiveProvider:
    """A trusted third party should use this class to generate crypto primitives."""

    _func_providers: Dict[str, Callable] = {}
    _ops_list: DefaultDict[str, List] = defaultdict(list)

    def __init__(self) -> None:  # noqa
        raise ValueError("This class should not be initialized")

    @staticmethod
    def generate_primitives(
        op_str: str,
        parties: List[Any],
        g_kwargs: Optional[Dict[str, Any]] = None,
        p_kwargs: Optional[Dict[str, Any]] = None,
        nr_instances: int = 1,
        ring_size: int = 2**32,
    ) -> List[Any]:
        g_kwargs = g_kwargs if g_kwargs is not None else {}
        p_kwargs = p_kwargs if p_kwargs is not None else {}
        """Generate "op_str" primitives.

        Args:
            op_str (str): Operator.
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

        nr_parties = len(parties)

        generator = CryptoPrimitiveProvider._func_providers[op_str]
        primitives = generator(
            **g_kwargs,
            nr_instances=nr_instances,
            nr_parties=nr_parties,
            ring_size=ring_size,
        )

        if p_kwargs is not None:
            """Do not transfer the primitives if there is not specified a
            values for populate kwargs."""
            CryptoPrimitiveProvider._transfer_primitives_to_parties(
                op_str=op_str,
                primitives=primitives,
                parties=parties,
                p_kwargs=p_kwargs,
                ring_size=ring_size,
            )

        return primitives

    @staticmethod
    def _transfer_primitives_to_parties(
        op_str: str,
        primitives: List[Any],
        parties: List[Any],
        p_kwargs: Dict[str, Any],
        ring_size: int,
    ) -> None:
        # relative
        from ...node.common.action.beaver_primitive_action import BeaverPrimitiveAction

        if not isinstance(primitives, list):
            raise ValueError("Primitives should be a List")

        if len(primitives) != len(parties):
            raise ValueError(
                f"Primitives length {len(primitives)} != Parties length {len(parties)}"
            )

        for primitives_party, client in zip(primitives, parties):

            args: List[Any] = []
            kwargs = {
                "op_str": op_str,
                "primitives": primitives_party,
                **p_kwargs,
                "ring_size": str(ring_size),
            }
            path = "syft.core.tensor.smpc.share_tensor.populate_store"
            cmd = BeaverPrimitiveAction(
                path=path,
                args=args,
                kwargs=kwargs,
                id_at_location=UID(),
                address=client.address,
            )
            client.send_immediate_msg_without_reply(msg=cmd)

    @staticmethod
    def get_state() -> str:
        """Get the state of a CryptoProvider.

        Returns:
            str: CryptoProvider
        """
        res = f"Providers: {list(CryptoPrimitiveProvider._func_providers.keys())}\n"
        return res
