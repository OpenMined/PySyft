# future
from __future__ import annotations

# stdlib
from collections import defaultdict
from enum import Enum
from typing import List
from typing import Optional
from typing import Set

# third party
from result import Err
from result import Ok
from result import Result

# relative
from ....core.node.common.node_table.syft_object import SyftObject
from ...common.serde.serializable import serializable
from .document_store import BaseStash
from .document_store import ClientConfig
from .document_store import DocumentStore
from .document_store import QueryKey
from .document_store import QueryKeys
from .document_store import StoreConfig
from .document_store import StorePartition
from .response import SyftSuccess


@serializable(recursive_serde=True)
class UniqueKeyCheck(Enum):
    EMPTY = 0
    MATCHES = 1
    ERROR = 2


@serializable(recursive_serde=True)
class DictStorePartition(StorePartition):
    def init_store(self, client_config: Optional[ClientConfig] = None) -> None:
        self.data = {}
        super().init_store()

        self.unique_keys = {}

        for partition_key in self.unique_cks:
            pk_key = partition_key.key
            self.unique_keys[pk_key] = {}

        self.searchable_keys = {}
        for partition_key in self.searchable_cks:
            pk_key = partition_key.key
            self.searchable_keys[pk_key] = defaultdict(list)

    def validate_partition_keys(
        self, store_query_key: QueryKey, unique_query_keys: QueryKeys
    ) -> UniqueKeyCheck:
        matches = []
        qks = unique_query_keys.all
        for qk in qks:
            pk_key, pk_value = qk.key, qk.value
            if pk_key not in self.unique_keys:
                raise Exception(
                    f"pk_key: {pk_key} not in unique_keys: {self.unique_keys.keys()}"
                )
            ck_col = self.unique_keys[pk_key]
            if pk_value in ck_col and ck_col[pk_value] == store_query_key.value:
                matches.append(pk_key)

        if len(matches) == 0:
            return UniqueKeyCheck.EMPTY
        elif len(matches) == len(qks):
            return UniqueKeyCheck.MATCHES

        return UniqueKeyCheck.ERROR

    def set_data_and_keys(
        self,
        store_query_key: QueryKey,
        unique_query_keys: QueryKeys,
        searchable_query_keys: QueryKeys,
        obj: SyftObject,
    ) -> None:
        # we should lock
        uqks = unique_query_keys.all
        for qk in uqks:
            pk_key, pk_value = qk.key, qk.value
            ck_col = self.unique_keys[pk_key]
            ck_col[pk_value] = store_query_key.value
            self.unique_keys[pk_key] = ck_col

        self.unique_keys[store_query_key.key][
            store_query_key.value
        ] = store_query_key.value

        sqks = searchable_query_keys.all
        for qk in sqks:
            pk_key, pk_value = qk.key, qk.value
            ck_col = self.searchable_keys[pk_key]
            ck_col[pk_value].append(store_query_key.value)
            self.searchable_keys[pk_key] = ck_col

        self.data[store_query_key.value] = obj

    def set(
        self,
        obj: SyftObject,
    ) -> Result[SyftObject, str]:
        try:
            store_query_key = self.settings.store_key.with_obj(obj)
            exists = store_query_key.value in self.data
            unique_query_keys = self.settings.unique_keys.with_obj(obj)
            searchable_query_keys = self.settings.searchable_keys.with_obj(obj)

            ck_check = self.validate_partition_keys(
                store_query_key=store_query_key, unique_query_keys=unique_query_keys
            )
            if not exists and ck_check == UniqueKeyCheck.EMPTY:
                self.set_data_and_keys(
                    store_query_key=store_query_key,
                    unique_query_keys=unique_query_keys,
                    searchable_query_keys=searchable_query_keys,
                    obj=obj,
                )
        except Exception as e:
            return Err(f"Failed to write obj {obj}. {e}")
        return Ok(obj)

    def all(self) -> Result[List[BaseStash.object_type], str]:
        return Ok(list(self.data.values()))

    def find_index_or_search_keys(
        self, index_qks: QueryKeys, search_qks: QueryKeys
    ) -> Result[List[SyftObject], str]:
        ids: Optional[Set] = None
        errors = []
        if len(index_qks.all) > 0:
            index_results = self._get_keys_index(qks=index_qks)
            if index_results.is_ok():
                if ids is None:
                    ids = index_results.ok()
                ids = ids.intersection(index_results.ok())
            else:
                errors.append(index_results.err())

        search_results = None
        if len(search_qks.all) > 0:
            search_results = self._find_keys_search(qks=search_qks)

            if search_results.is_ok():
                if ids is None:
                    ids = search_results.ok()
                ids = ids.intersection(search_results.ok())
            else:
                errors.append(search_results.err())

        if len(errors) > 0:
            return Err(" ".join(errors))

        qks = self.store_query_keys(ids)
        return self.get_all_from_store(qks=qks)

    def remove_keys(
        self,
        unique_query_keys: QueryKeys,
        searchable_query_keys: QueryKeys,
    ) -> None:
        uqks = unique_query_keys.all
        for qk in uqks:
            pk_key, pk_value = qk.key, qk.value
            ck_col = self.unique_keys[pk_key]
            ck_col.pop(pk_value, None)

        sqks = searchable_query_keys.all
        for qk in sqks:
            pk_key, pk_value = qk.key, qk.value
            ck_col = self.searchable_keys[pk_key]
            ck_col.pop(pk_value, None)

    def update(self, qk: QueryKey, obj: SyftObject) -> Result[SyftObject, str]:
        try:
            if qk.value not in self.data:
                return Err(f"No object exists for query key: {qk}")

            _original_obj = self.data[qk.value]
            _original_unique_keys = self.settings.unique_keys.with_obj(_original_obj)
            _original_searchable_keys = self.settings.searchable_keys.with_obj(
                _original_obj
            )

            # 🟡 TODO 28: Add locking in this transaction

            # remove old keys
            self.remove_keys(
                unique_query_keys=_original_unique_keys,
                searchable_query_keys=_original_searchable_keys,
            )

            # update the object with new data
            for key, value in obj.dict(exclude_none=True).items():
                setattr(_original_obj, key, value)

            # update data and keys
            self.set_data_and_keys(
                store_query_key=qk,
                unique_query_keys=self.settings.unique_keys.with_obj(_original_obj),
                searchable_query_keys=self.settings.searchable_keys.with_obj(
                    _original_obj
                ),
                obj=_original_obj,
            )

            return Ok(_original_obj)
        except Exception as e:
            return Err(f"Failed to update obj {obj} with error: {e}")

    def get_all_from_store(self, qks: QueryKeys) -> Result[List[SyftObject], str]:
        matches = []
        for qk in qks.all:
            if qk.value in self.data:
                matches.append(self.data[qk.value])
        return Ok(matches)

    def _delete_unique_keys_for(self, obj: SyftObject) -> Result[SyftSuccess, str]:
        for _unique_ck in self.unique_cks:
            qk = _unique_ck.with_obj(obj)
            self.unique_keys[qk.key].pop(qk.value, None)
        return Ok(SyftSuccess(message="Deleted"))

    def _delete_search_keys_for(self, obj: SyftObject) -> Result[SyftSuccess, str]:
        for _search_ck in self.searchable_cks:
            qk = _search_ck.with_obj(obj)
            self.searchable_keys[qk.key].pop(qk.value, None)
        return Ok(SyftSuccess(message="Deleted"))

    def _get_keys_index(self, qks: QueryKeys) -> Result[Set[QueryKey], str]:
        try:
            # match AND
            subsets = []
            for qk in qks.all:
                subset = {}
                pk_key, pk_value = qk.key, qk.value
                if pk_key not in self.unique_keys:
                    return Err(f"Failed to query index with {qk}")
                ck_col = self.unique_keys[pk_key]
                if pk_value not in ck_col.keys():
                    # must be at least one in all query keys
                    continue
                store_value = ck_col[pk_value]
                subsets.append({store_value})

            if len(subsets) == 0:
                return Ok(set())
            # AND
            subset = subsets.pop()
            for s in subsets:
                subset = subset.intersection(s)

            return Ok(subset)
        except Exception as e:
            return Err(f"Failed to query with {qks}. {e}")

    def _find_keys_search(self, qks: QueryKeys) -> Result[Set[QueryKey], str]:
        try:
            # match AND
            subsets = []
            for qk in qks.all:
                subset = {}
                pk_key, pk_value = qk.key, qk.value
                if pk_key not in self.searchable_keys:
                    return Err(f"Failed to search with {qk}")
                ck_col = self.searchable_keys[pk_key]
                if pk_value not in ck_col.keys():
                    # must be at least one in all query keys
                    continue
                store_values = ck_col[pk_value]
                subsets.append(set(store_values))

            if len(subsets) == 0:
                return Ok(set())
            # AND
            subset = subsets.pop()
            for s in subsets:
                subset = subset.intersection(s)

            return Ok(subset)
        except Exception as e:
            return Err(f"Failed to query with {qks}. {e}")

    def create(self, obj: SyftObject) -> Result[SyftObject, str]:
        pass

    def delete(self, qk: QueryKey) -> Result[SyftSuccess, Err]:
        try:
            _obj = self.data.pop(qk.value)
            self._delete_unique_keys_for(_obj)
            self._delete_search_keys_for(_obj)
            return Ok(SyftSuccess(message="Deleted"))
        except Exception as e:
            return Err(f"Failed to delete with query key {qk} with error: {e}")


# the base document store is already a dict but we can change it later
@serializable(recursive_serde=True)
class DictDocumentStore(DocumentStore):
    partition_type = DictStorePartition


DictStoreConfig = StoreConfig(store_type=DictDocumentStore)
