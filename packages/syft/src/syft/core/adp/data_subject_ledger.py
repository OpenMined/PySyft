# future
from __future__ import annotations

# stdlib
import time
from typing import Callable
from typing import Final
from typing import Optional
from typing import Tuple

# third party
from nacl.signing import VerifyKey
import numpy as np
from scipy.optimize import minimize_scalar

# syft absolute
import syft as sy

# relative
from ..common.serde.serializable import serializable
from ..store.store_interface import StoreKey


class LedgerUpdate:
    def __init__(
        self,
        sigmas: np.ndarray,
        l2_norms: np.ndarray,
        l2_norm_bounds: np.ndarray,
        Ls: np.ndarray,
        coeffs: np.ndarray,
        entity_ids: np.ndarray,
        update_number: int,
        timestamp: Optional[float] = None,
    ):
        self.sigmas = sigmas
        self.l2_norms = l2_norms
        self.l2_norm_bounds = l2_norm_bounds
        self.Ls = Ls
        self.coeffs = coeffs
        self.entity_ids = entity_ids
        self.update_number = update_number
        self.timestamp = timestamp if timestamp is not None else time.time()


class LedgerStore:
    def get(self, key: StoreKey) -> DataSubjectLedger:
        raise NotImplementedError

    def set(self, key: StoreKey, value: DataSubjectLedger) -> None:
        raise NotImplementedError


class DictLedgerStore(LedgerStore):
    def __init__(self) -> None:
        self.kv_store = {}

    def get(self, key: StoreKey) -> DataSubjectLedger:
        return self.kv_store[key]

    def set(self, key: StoreKey, value: DataSubjectLedger) -> None:
        self.kv_store[key] = value


@serializable(recursive_serde=True)
class DataSubjectLedger:
    """for a particular data subject, this is the list
    of all mechanisms releasing informationo about this
    particular subject, stored in a vectorized form"""

    __attr_allowlist__ = [
        "_entity_ids_query",
        "_rdp_constants",
        "timestamp",
        "timestamp_of_last_update",
        "update_number",
    ]

    def __init__(
        self, store: LedgerStore, user_key: VerifyKey, default_cache_size: int = 1_000
    ) -> None:
        self.store = store
        self.user_key = user_key
        self.sigmas = np.array([])
        self.l2_norms = np.array([])
        self.l2_norm_bounds = np.array([])
        self.Ls = np.array([])
        self.coeffs = np.array([])
        self.entity_ids = np.array([])
        self.entity2budget = np.array([])

        # self.reset()
        self.cache_constant2epsilon = np.array([])
        self.increase_max_cache(int(default_cache_size))

        # save initial size (number of rows from DB) when deserialized
        # self.known_db_size = 0
        self.update_number = 0
        self.timestamp_of_last_update: Optional[float] = None
        self.timestamp = time.time()

        # our cached query
        self._updated_entity_ids_query: bool = False
        self._updated_rdp_constants: bool = False
        self.entity_ids_query = np.array([])
        self.rdp_constants = np.array([])

    @property
    def delta(self) -> float:
        FIXED_DELTA: Final = 1e-6
        return FIXED_DELTA  # WARNING: CHANGING DELTA INVALIDATES THE CACHE

    def write_to_db(self):
        self.update_number += 1
        print("saving to db", len(self.entity_ids_query), len(self.rdp_constants))
        print("data", self.entity_ids_query, self.rdp_constants)
        ser = sy.serialize(self, to_bytes=True)
        try:
            # todo change user_key or uid?
            self.store.set(key=self.user_key, value=ser)

        except Exception as e:
            print(f"Failed to write ledger to ledger store. {e}")
            raise e

        self._updated_entity_ids_query = False
        self._updated_rdp_constants = False

        # result = LedgerUpdate(
        #     sigmas=self.sigmas[self.known_db_size :],  # noqa: E203
        #     l2_norms=self.l2_norms[self.known_db_size :],  # noqa: E203
        #     l2_norm_bounds=self.l2_norms[self.known_db_size :],  # noqa: E203
        #     Ls=self.Ls[self.known_db_size :],  # noqa: E203
        #     coeffs=self.coeffs[self.known_db_size :],  # noqa: E203
        #     entity_ids=self.entity_ids[self.known_db_size :],  # noqa: E203
        #     update_number=self.update_number,
        #     timestamp=time.time(),
        # )
        # self.known_db_size += len(self.sigmas)
        # return result

    @staticmethod
    def get_or_create(store: LedgerStore, user_key: VerifyKey) -> None:
        ledger: Optional[DataSubjectLedger] = None
        try:
            # todo change user_key or uid?
            buf = store.get(key=user_key)
            ledger = sy.deserialize(buf, from_bytes=True)
            ledger.store = store
            ledger.user_key = user_key
        except KeyError:
            print("Creating new Ledger")
            ledger = DataSubjectLedger(store=store, user_key=user_key)
        except Exception as e:
            print(f"Failed to read ledger from ledger store. {e}")
        return ledger

        # self._updated_entity_ids_query = False
        # self._updated_rdp_constants = False

        # if update.update_number == self.update_number + 1:
        #     if (
        #         self.timestamp_of_last_update is not None
        #         and update.timestamp is not None
        #         and update.timestamp < self.timestamp
        #     ):
        #         raise Exception(
        #             "It appears that updates were created out of order."
        #             + "This is probably due to multiple python threads creating updates"
        #             + "which should NOT happen. This is a very serious error. "
        #             + "Please contact OpenMined immediately. Thank you!"
        #         )
        #     self.sigmas = np.concatenate([self.sigmas, update.sigmas])
        #     self.l2_norms = np.concatenate([self.l2_norms, update.l2_norms])
        #     self.l2_norm_bounds = np.concatenate(
        #         [self.l2_norm_bounds, update.l2_norm_bounds]
        #     )
        #     self.Ls = np.concatenate([self.Ls, update.Ls])
        #     self.coeffs = np.concatenate([self.coeffs, update.coeffs])
        #     self.entity_ids = np.concatenate([self.entity_ids, update.entity_ids])
        #     self.update_number = update.update_number
        #     self.timestamp = update.timestamp
        # else:
        #     raise Exception("Cannot add update to Ledger")

    # def reset(self) -> None:
    #     self.sigmas = np.array([])
    #     self.l2_norms = np.array([])
    #     self.l2_norm_bounds = np.array([])
    #     self.Ls = np.array([])
    #     self.coeffs = np.array([])
    #     self.entity_ids = np.array([])
    #     self.entity2budget = np.array([])

    def batch_append(
        self,
        sigmas: np.ndarray,
        l2_norms: np.ndarray,
        l2_norm_bounds: np.ndarray,
        Ls: np.ndarray,
        coeffs: np.ndarray,
        entity_ids: np.ndarray,
    ) -> None:
        self.sigmas = np.concatenate([self.sigmas, sigmas.flatten()])
        print(type(self.l2_norms), self.l2_norms, self.l2_norms.shape)
        print(type(l2_norms), l2_norms, l2_norms.shape)
        print("fdsa")
        self.l2_norms = np.concatenate([self.l2_norms, l2_norms.flatten()])
        self.l2_norm_bounds = np.concatenate(
            [self.l2_norm_bounds, l2_norm_bounds.flatten()]
        )
        self.Ls = np.concatenate([self.Ls, Ls.flatten()])
        self.coeffs = np.concatenate([self.coeffs, coeffs.flatten()])
        self.entity_ids = np.concatenate([self.entity_ids, entity_ids.flatten()])

    def increase_max_cache(self, new_size: int) -> None:
        new_entries = []
        current_size = len(self.cache_constant2epsilon)
        for i in range(new_size - current_size):
            alpha, eps = self.get_optimal_alpha_for_constant(
                constant=i + 1 + current_size
            )
            new_entries.append(eps)
        self.cache_constant2epsilon = np.concatenate(
            [self.cache_constant2epsilon, np.array(new_entries)]
        )
        # print(self.cache_constant2epsilon)

    def get_fake_rdp_func(self, constant: int) -> Callable:
        def func(alpha: float) -> float:
            return alpha * constant

        return func

    def get_alpha_search_function(self, rdp_compose_func: Callable) -> Callable:
        # if len(self.deltas) > 0:
        # delta = np.max(self.deltas)
        # else:
        log_delta = np.log(self.delta)

        def fun(alpha: float) -> float:  # the input is the RDP's \alpha
            if alpha <= 1:
                return np.inf
            else:
                alpha_minus_1 = alpha - 1
                return np.maximum(
                    rdp_compose_func(alpha)
                    + np.log(alpha_minus_1 / alpha)
                    - (log_delta + np.log(alpha)) / alpha_minus_1,
                    0,
                )

        return fun

    def get_optimal_alpha_for_constant(
        self, constant: int = 3
    ) -> Tuple[np.ndarray, Callable]:
        f = self.get_fake_rdp_func(constant=constant)
        f2 = self.get_alpha_search_function(rdp_compose_func=f)
        results = minimize_scalar(
            f2, method="Brent", bracket=(1, 2), bounds=[1, np.inf]
        )

        return results.x, results.fun

    def get_batch_rdp_constants(
        self, entity_ids_query: np.ndarray, private: bool = True
    ) -> np.ndarray:
        if self._updated_entity_ids_query or self._updated_rdp_constants:
            raise Exception(
                f"We need to save the DataSubjectLedger before querying again"
            )
        # get indices for all ledger rows corresponding to any of the entities in
        # entity_ids_query
        indices_batch = np.where(np.in1d(self.entity_ids, entity_ids_query))[0]

        # use the indices to get a "batch" of the full ledger. this is the only part
        # of the ledger we care about (the entries corresponding to specific entities)
        batch_sigmas = self.sigmas.take(indices_batch)
        batch_Ls = self.Ls.take(indices_batch)
        batch_l2_norms = self.l2_norms.take(indices_batch)
        batch_l2_norm_bounds = self.l2_norm_bounds.take(indices_batch)
        batch_coeffs = self.coeffs.take(indices_batch)
        batch_entity_ids = self.entity_ids.take(indices_batch).astype(np.int64)

        squared_Ls = batch_Ls ** 2
        squared_sigma = batch_sigmas ** 2

        if private:
            squared_L2_norms = batch_l2_norms ** 2
            constant = (
                squared_Ls * squared_L2_norms / (2 * squared_sigma)
            ) * batch_coeffs
            constant = np.bincount(batch_entity_ids, weights=constant).take(
                entity_ids_query
            )
        else:
            squared_L2_norm_bounds = batch_l2_norm_bounds ** 2
            constant = (
                squared_Ls * squared_L2_norm_bounds / (2 * squared_sigma)
            ) * batch_coeffs
            constant = np.bincount(batch_entity_ids, weights=constant).take(
                entity_ids_query
            )

        self._updated_rdp_constants = True
        self._updated_entity_ids_query = True
        self.rdp_constants = np.concatenate([self.rdp_constants, constant])
        self.entity_ids_query = np.concatenate(
            [self.entity_ids_query, entity_ids_query]
        )
        return constant

    def get_epsilon_spend(self, entity_ids_query: np.ndarray) -> np.ndarray:
        rdp_constants = self.get_batch_rdp_constants(
            entity_ids_query=entity_ids_query
        ).astype(np.int64)
        rdp_constants_lookup = rdp_constants - 1
        try:
            eps_spend = self.cache_constant2epsilon.take(rdp_constants_lookup)
        except IndexError:
            self.increase_max_cache(int(max(rdp_constants_lookup) * 1.1))
            eps_spend = self.cache_constant2epsilon.take(rdp_constants_lookup)
        return eps_spend

    def get_overbudgeted_entities(
        self, user_budget: float, unique_entity_ids_query: np.ndarray
    ) -> np.ndarray:
        """TODO:
        In our current implementation, user_budget is obtained by querying the
        Adversarial Accountant's entity2ledger with the Data Scientist's User Key.
        When we replace the entity2ledger with something else, we could perhaps directly
        add it into this method
        """

        # Get the privacy budget spent by all the entities
        epsilon_spent = self.get_epsilon_spend(unique_entity_ids_query.astype(np.int64))
        # print(np.mean(epsilon_spent))

        # Create a mask
        is_overbudget = np.ones_like(epsilon_spent) * user_budget < epsilon_spent
        return is_overbudget
