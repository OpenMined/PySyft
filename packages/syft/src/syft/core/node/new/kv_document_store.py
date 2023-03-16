# future
from __future__ import annotations

# stdlib
from collections import defaultdict
from enum import Enum
from typing import Any
from typing import List
from typing import Optional
from typing import Set

# third party
from result import Err
from result import Ok
from result import Result
from typing_extensions import Self

# relative
from .document_store import BaseStash
from .document_store import QueryKey
from .document_store import QueryKeys
from .document_store import StorePartition
from .response import SyftSuccess
from .serializable import serializable
from .syft_object import SyftObject


@serializable(recursive_serde=True)
class UniqueKeyCheck(Enum):
    EMPTY = 0
    MATCHES = 1
    ERROR = 2


class KeyValueBackingStore:
    def __setitem__(self, key: Any, value: Any) -> None:
        raise NotImplementedError

    def __getitem__(self, key: Any) -> Self:
        raise NotImplementedError

    def __repr__(self) -> str:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __delitem__(self, key: str):
        raise NotImplementedError

    def clear(self) -> Self:
        raise NotImplementedError

    def copy(self) -> Self:
        raise NotImplementedError

    def update(self, *args: Any, **kwargs: Any) -> Self:
        raise NotImplementedError

    def keys(self) -> Any:
        raise NotImplementedError

    def values(self) -> Any:
        raise NotImplementedError

    def items(self) -> Any:
        raise NotImplementedError

    def pop(self, *args: Any) -> Self:
        raise NotImplementedError

    def __contains__(self, item: Any) -> bool:
        raise NotImplementedError

    def __iter__(self) -> Any:
        raise NotImplementedError


class KeyValueStorePartition(StorePartition):
    def init_store(self) -> None:
        super().init_store()
        self.data = self.store_config.backing_store(
            "data", self.settings, self.store_config
        )
        self.unique_keys = self.store_config.backing_store(
            "unique_keys", self.settings, self.store_config
        )
        self.searchable_keys = self.store_config.backing_store(
            "searchable_keys", self.settings, self.store_config
        )

        for partition_key in self.unique_cks:
            pk_key = partition_key.key
            if pk_key not in self.unique_keys:
                self.unique_keys[pk_key] = {}

        for partition_key in self.searchable_cks:
            pk_key = partition_key.key
            if pk_key not in self.searchable_keys:
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
            if pk_value in ck_col or ck_col.get(pk_value) == store_query_key.value:
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
            if qk.type_list:
                # coerce the list of objects to strings for a single key
                pk_value = " ".join([str(obj) for obj in pk_value])

            ck_col[pk_value].append(store_query_key.value)
            self.searchable_keys[pk_key] = ck_col

        self.data[store_query_key.value] = obj

    def set(
        self, obj: SyftObject, ignore_duplicates: bool = False
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
            elif not ignore_duplicates:
                return Err(f"Duplication Key Error: {obj}")
        except Exception as e:
            return Err(f"Failed to write obj {obj}. {e}")
        return Ok(obj)

    def all(self) -> Result[List[BaseStash.object_type], str]:
        return Ok(list(self.data.values()))

    def __len__(self) -> Result[List[BaseStash.object_type], str]:
        return len(self.data)

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
            for key, value in obj.to_dict(exclude_none=True).items():
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
                if qk.type_list:
                    # 🟡 TODO: change this hacky way to do on to many relationships
                    # this is when you search a QueryKey which is a list of items
                    # at the moment its mostly just a List[UID]
                    # match OR against all keys for this col
                    # the values of the list will be turned into strings in a single key
                    matches = set()
                    for item in pk_value:
                        for col_key in ck_col.keys():
                            if str(item) in col_key:
                                store_values = ck_col[col_key]
                                for value in store_values:
                                    matches.add(value)
                    if len(matches):
                        subsets.append(matches)
                else:
                    # this is the normal path
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
