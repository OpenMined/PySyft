# future
from __future__ import annotations

# stdlib
import time
from typing import Any
from typing import Callable
from typing import Final
from typing import Optional
from typing import Tuple

# third party
from nacl.signing import VerifyKey
import numpy as np
from scipy.optimize import minimize_scalar

# relative
from ...core.node.common.node_manager.user_manager import RefreshBudgetException
from ..common.serde.serializable import serializable
from .abstract_ledger_store import AbstractDataSubjectLedger
from .abstract_ledger_store import AbstractLedgerStore


class RDPParams:
    def __init__(
        self,
        sigmas: np.ndarray,
        l2_norms: np.ndarray,
        l2_norm_bounds: np.ndarray,
        Ls: np.ndarray,
        coeffs: np.ndarray,
    ) -> None:
        self.sigmas = sigmas
        self.l2_norms = l2_norms
        self.l2_norm_bounds = l2_norm_bounds
        self.Ls = Ls
        self.coeffs = coeffs


@serializable(recursive_serde=True)
class DataSubjectLedger(AbstractDataSubjectLedger):
    """for a particular data subject, this is the list
    of all mechanisms releasing informationo about this
    particular subject, stored in a vectorized form"""

    __attr_allowlist__ = [
        "_update_number",
        "_timestamp_of_last_update",
        "_entity_ids_query",
        "_rdp_constants",
        "_pending_save",  # theres no init on recursive serde so this isnt set to False
    ]

    def __init__(
        self,
        store: AbstractLedgerStore,
        user_key: VerifyKey,
        default_cache_size: int = 1_000,
    ) -> None:
        self.store = store
        self.user_key = user_key

        # where is this absolute list coming from?
        self.entity_ids = np.array([], dtype=np.int64)

        self._cache_constant2epsilon = np.array([], dtype=np.float64)
        self._increase_max_cache(int(default_cache_size))

        # tracking atomic updates
        self._pending_save: bool = False
        self._update_number = 0
        self._timestamp_of_last_update = time.time()
        self._entity_ids_query = np.array([], dtype=np.int64)
        self._rdp_constants = np.array([], dtype=np.float64)

    @property
    def delta(self) -> float:
        FIXED_DELTA: Final = 1e-6
        return FIXED_DELTA  # WARNING: CHANGING DELTA INVALIDATES THE CACHE

    @staticmethod
    def get_or_create(
        store: AbstractLedgerStore, user_key: VerifyKey
    ) -> Optional[AbstractDataSubjectLedger]:
        ledger: Optional[AbstractDataSubjectLedger] = None
        try:
            # todo change user_key or uid?
            ledger = store.get(key=user_key)
            ledger.store = store
            ledger.user_key = user_key
        except KeyError:
            print("Creating new Ledger")
            ledger = DataSubjectLedger(store=store, user_key=user_key)
        except Exception as e:
            print(f"Failed to read ledger from ledger store. {e}")
        return ledger

    def get_entity_overbudget_mask_for_epsilon_and_append(
        self,
        unique_entity_ids_query: np.ndarray,
        rdp_params: RDPParams,
        node: Any,
        private: bool = True,
    ) -> np.ndarray:
        # coerce to np.int64
        entity_ids_query: np.ndarray = unique_entity_ids_query.astype(np.int64)

        # calculate constants
        rdp_constants = self._get_batch_rdp_constants(
            entity_ids_query=entity_ids_query, rdp_params=rdp_params, private=private
        )

        epsilon_spent, mask = self._get_overbudgeted_entities(
            node=node,
            rdp_constants=rdp_constants,
        )

        if self._write_ledger(node=node, epsilon_spent=epsilon_spent):
            return mask

    def _write_ledger(self, node: Any, epsilon_spent: float) -> bool:
        print("write epsilon in transaction with serializing the ledger", epsilon_spent)
        self._update_number += 1
        try:
            self._pending_save = False
            self.store.set(key=self.user_key, value=self)
            return True
        except Exception as e:
            self._pending_save = True
            print(f"Failed to write ledger to ledger store. {e}")
            raise e

    def _increase_max_cache(self, new_size: int) -> None:
        new_entries = []
        current_size = len(self._cache_constant2epsilon)
        for i in range(new_size - current_size):
            _, eps = self._get_optimal_alpha_for_constant(constant=i + 1 + current_size)
            new_entries.append(eps)
        self._cache_constant2epsilon = np.concatenate(
            [self._cache_constant2epsilon, np.array(new_entries)]
        )

    def _get_fake_rdp_func(self, constant: int) -> Callable:
        def func(alpha: float) -> float:
            return alpha * constant

        return func

    def _get_alpha_search_function(self, rdp_compose_func: Callable) -> Callable:
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

    def _get_optimal_alpha_for_constant(
        self, constant: int = 3
    ) -> Tuple[np.ndarray, Callable]:
        f = self._get_fake_rdp_func(constant=constant)
        f2 = self._get_alpha_search_function(rdp_compose_func=f)
        results = minimize_scalar(
            f2, method="Brent", bracket=(1, 2), bounds=[1, np.inf]
        )

        return results.x, results.fun

    def _get_batch_rdp_constants(
        self, entity_ids_query: np.ndarray, rdp_params: RDPParams, private: bool = True
    ) -> np.ndarray:
        if self._pending_save:
            raise Exception(
                "We need to save the DataSubjectLedger before querying again"
            )
        self._pending_save = True
        # get indices for all ledger rows corresponding to any of the entities in
        # entity_ids_query
        indices_batch = np.where(np.in1d(self.entity_ids, entity_ids_query))[0]

        # use the indices to get a "batch" of the full ledger. this is the only part
        # of the ledger we care about (the entries corresponding to specific entities)
        batch_sigmas = rdp_params.sigmas.take(indices_batch)
        batch_l2_norms = rdp_params.l2_norms.take(indices_batch)
        batch_Ls = rdp_params.Ls.take(indices_batch)
        batch_l2_norm_bounds = rdp_params.l2_norm_bounds.take(indices_batch)
        batch_coeffs = rdp_params.coeffs.take(indices_batch)

        batch_entity_ids = self.entity_ids.take(indices_batch)

        squared_Ls = batch_Ls**2
        squared_sigma = batch_sigmas**2

        if private:
            squared_L2_norms = batch_l2_norms**2
            constant = (
                squared_Ls * squared_L2_norms / (2 * squared_sigma)
            ) * batch_coeffs
            constant = np.bincount(batch_entity_ids, weights=constant).take(
                entity_ids_query
            )
        else:
            squared_L2_norm_bounds = batch_l2_norm_bounds**2
            constant = (
                squared_Ls * squared_L2_norm_bounds / (2 * squared_sigma)
            ) * batch_coeffs
            constant = np.bincount(batch_entity_ids, weights=constant).take(
                entity_ids_query
            )

        # update our serialized format with the calculated constants
        self._rdp_constants = np.concatenate([self._rdp_constants, constant])
        self._entity_ids_query = np.concatenate(
            [self._entity_ids_query, entity_ids_query]
        )
        return constant

    def _get_epsilon_spend(self, rdp_constants: np.ndarray) -> np.ndarray:
        rdp_constants_lookup = (rdp_constants - 1).astype(np.int64)
        # TODO remove this hack
        if not hasattr(self, "_cache_constant2epsilon"):
            self._cache_constant2epsilon = np.array([], dtype=np.float64)
            self._increase_max_cache(1_000)
        try:
            # needed as np.int64 to use take
            eps_spend = self._cache_constant2epsilon.take(rdp_constants_lookup)
        except IndexError:
            self._increase_max_cache(int(max(rdp_constants_lookup) * 1.1))
            eps_spend = self._cache_constant2epsilon.take(rdp_constants_lookup)
        return eps_spend

    def _get_overbudgeted_entities(
        self,
        node: Any,
        rdp_constants: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        """TODO:
        In our current implementation, user_budget is obtained by querying the
        Adversarial Accountant's entity2ledger with the Data Scientist's User Key.
        When we replace the entity2ledger with something else, we could perhaps directly
        add it into this method
        """

        # Get the privacy budget spent by all the entities
        epsilon_spent = self._get_epsilon_spend(rdp_constants=rdp_constants)
        # print(np.mean(epsilon_spent))
        user_epsilon_spend = max(epsilon_spent)

        attempts = 0
        user_budget = None
        while attempts < 5:
            print(f"Attemping to spend epsilon: {user_epsilon_spend}. Try: {attempts}")
            attempts += 1
            try:
                user_budget = self.spend_epsilon(
                    node=node, epsilon_spend=user_epsilon_spend
                )
                break
            except RefreshBudgetException as e:
                # this is the only exception we allow to retry
                print("got refresh budget error", e)
            except Exception as e:
                print(f"Problem spending epsilon. {e}")
                raise e

        if user_budget is None:
            raise Exception("Failed to spend_epsilon")

        # Create a mask
        return (
            epsilon_spent,
            np.ones_like(epsilon_spent) * user_budget < epsilon_spent,
        )

    def spend_epsilon(self, node: Any, epsilon_spend: float) -> bool:
        # get the budget
        user_budget = node.users.get_budget_for_user(verify_key=self.user_key)
        print("got user budget", user_budget, "epsilon_spent", epsilon_spend)
        node.users.deduct_epsilon_for_user(
            verify_key=self.user_key,
            old_budget=user_budget,
            epsilon_spend=epsilon_spend,
        )
        # return the budget we used
        return user_budget
