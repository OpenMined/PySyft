# stdlib
from collections.abc import Callable
from collections.abc import Iterable
from enum import Enum
from functools import partial
import os
from pathlib import Path
import tempfile
from typing import Any

# third party
import matplotlib.pyplot as plt
import networkx as nx
from pydantic import Field
from pydantic import field_validator
from result import Err
from result import Ok
from result import Result
from typing_extensions import Self

# relative
from ...node.credentials import SyftVerifyKey
from ...serde.deserialize import _deserialize
from ...serde.serializable import serializable
from ...serde.serialize import _serialize
from ...store.document_store import QueryKey
from ...store.document_store import QueryKeys
from ...store.document_store import StoreClientConfig
from ...store.document_store import StoreConfig
from ...store.locks import LockingConfig
from ...store.locks import SyftLock
from ...store.locks import ThreadingLockingConfig
from ...types.datetime import DateTime
from ...types.syft_object import PartialSyftObject
from ...types.syft_object import SYFT_OBJECT_VERSION_2
from ...types.syft_object import SyftObject
from ...types.uid import UID
from .action_object import Action
from .action_object import ActionObject


@serializable()
class ExecutionStatus(Enum):
    PROCESSING = 0
    DONE = 1
    FAILED = 2


@serializable()
class NodeType(Enum):
    ACTION = Action
    ACTION_OBJECT = ActionObject


@serializable()
class NodeActionData(SyftObject):
    __canonical_name__ = "NodeActionData"
    __version__ = SYFT_OBJECT_VERSION_2

    id: UID | None = None  # type: ignore[assignment]
    type: NodeType
    status: ExecutionStatus = ExecutionStatus.PROCESSING
    retry: int = 0
    created_at: DateTime = Field(default_factory=DateTime.now)
    updated_at: DateTime = Field(default_factory=DateTime.now)
    user_verify_key: SyftVerifyKey
    is_mutated: bool = False  # denotes that this node has been mutated
    is_mutagen: bool = False  # denotes that this node is causing a mutation
    next_mutagen_node: UID | None = None  # next neighboring mutagen node
    last_nm_mutagen_node: UID | None = None  # last non mutated mutagen node

    @classmethod
    def from_action(cls, action: Action, credentials: SyftVerifyKey) -> Self:
        is_mutagen = action.remote_self is not None and (
            action.remote_self == action.result_id
        )
        return cls(
            id=action.id,
            type=NodeType.ACTION,
            user_verify_key=credentials,
            is_mutagen=is_mutagen,
        )

    @classmethod
    def from_action_obj(
        cls, action_obj: ActionObject, credentials: SyftVerifyKey
    ) -> Self:
        return cls(
            id=action_obj.id,
            type=NodeType.ACTION_OBJECT,
            user_verify_key=credentials,
        )

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, NodeActionData):
            raise NotImplementedError(
                "Comparisions can be made with NodeActionData type objects only."
            )
        return hash(self) == hash(other)

    def __repr__(self) -> str:
        return self._repr_debug_()


@serializable()
class NodeActionDataUpdate(PartialSyftObject):
    __canonical_name__ = "NodeActionDataUpdate"
    __version__ = SYFT_OBJECT_VERSION_2

    id: UID
    type: NodeType
    status: ExecutionStatus
    retry: int
    created_at: DateTime
    updated_at: DateTime = Field(default_factory=DateTime.now)
    credentials: SyftVerifyKey
    is_mutated: bool
    is_mutagen: bool
    next_mutagen_node: UID  # next neighboring mutagen node
    last_nm_mutagen_node: UID  # last non mutated mutagen node


@serializable()
class BaseGraphStore:
    graph_type: Any
    client_config: StoreClientConfig | None

    def set(self, uid: Any, data: Any) -> None:
        raise NotImplementedError

    def get(self, uid: Any) -> Any:
        raise NotImplementedError

    def delete(self, uid: Any) -> None:
        raise NotImplementedError

    def find_neighbors(self, uid: Any) -> list | None:
        raise NotImplementedError

    def update(self, uid: Any, data: Any) -> None:
        raise NotImplementedError

    def add_edge(self, parent: Any, child: Any) -> None:
        raise NotImplementedError

    def remove_edge(self, parent: Any, child: Any) -> None:
        raise NotImplementedError

    def nodes(self) -> Any:
        raise NotImplementedError

    def edges(self) -> Any:
        raise NotImplementedError

    def visualize(self, seed: int, figsize: tuple) -> None:
        raise NotImplementedError

    def save(self) -> None:
        raise NotImplementedError

    def get_predecessors(self, uid: UID) -> list:
        raise NotImplementedError

    def get_successors(self, uid: UID) -> list:
        raise NotImplementedError

    def exists(self, uid: Any) -> bool:
        raise NotImplementedError

    def subgraph(self, qks: QueryKeys) -> Any:
        raise NotImplementedError

    def topological_sort(self, subgraph: Any) -> Any:
        raise NotImplementedError


@serializable()
class InMemoryStoreClientConfig(StoreClientConfig):
    filename: str = "action_graph.bytes"
    path: str | Path = Field(default_factory=tempfile.gettempdir)

    # We need this in addition to Field(default_factory=...)
    # so users can still do InMemoryStoreClientConfig(path=None)
    @field_validator("path", mode="before")
    @classmethod
    def __default_path(cls, path: str | Path | None) -> str | Path:
        if path is None:
            return tempfile.gettempdir()
        return path

    @property
    def file_path(self) -> Path:
        return Path(self.path) / self.filename


@serializable(without=["_lock"])
class NetworkXBackingStore(BaseGraphStore):
    def __init__(self, store_config: StoreConfig, reset: bool = False) -> None:
        if store_config.client_config:
            self.path_str = store_config.client_config.file_path.as_posix()
        else:
            self.path_str = ""
        if not reset and os.path.exists(self.path_str):
            self._db = self._load_from_path(self.path_str)
        else:
            self._db = nx.DiGraph()

        self.locking_config = store_config.locking_config
        self._lock: SyftLock | None = None

    @property
    def lock(self) -> SyftLock:
        if not hasattr(self, "_lock") or self._lock is None:
            self._lock = SyftLock(self.locking_config)
        return self._lock

    @property
    def db(self) -> nx.Graph:
        return self._db

    def _thread_safe_cbk(
        self, cbk: Callable, *args: Any, **kwargs: Any
    ) -> Result[Any, str]:
        # TODO copied method from document_store, have it in one place and reuse?
        locked = self.lock.acquire(blocking=True)
        if not locked:
            return Err(
                f"Failed to acquire lock for the operation {self.lock.lock_name} ({self.lock._lock})"
            )
        try:
            result = cbk(*args, **kwargs)
        except BaseException as e:
            result = Err(str(e))
        self.lock.release()

        return result

    def set(self, uid: UID, data: Any) -> None:
        self._thread_safe_cbk(self._set, uid=uid, data=data)

    def _set(self, uid: UID, data: Any) -> None:
        if self.exists(uid=uid):
            self.update(uid=uid, data=data)
        else:
            self.db.add_node(uid, data=data)
        self.save()

    def get(self, uid: UID) -> Any:
        node_data = self.db.nodes.get(uid)
        return node_data.get("data")

    def exists(self, uid: Any) -> bool:
        return uid in self.nodes()

    def delete(self, uid: UID) -> None:
        self._thread_safe_cbk(self._delete, uid=uid)

    def _delete(self, uid: UID) -> None:
        if self.exists(uid=uid):
            self.db.remove_node(uid)
        self.save()

    def find_neighbors(self, uid: UID) -> list | None:
        if self.exists(uid=uid):
            neighbors = self.db.neighbors(uid)
            return neighbors
        return None

    def update(self, uid: UID, data: Any) -> None:
        self._thread_safe_cbk(self._update, uid=uid, data=data)

    def _update(self, uid: UID, data: Any) -> None:
        if self.exists(uid=uid):
            self.db.nodes[uid]["data"] = data
        self.save()

    def add_edge(self, parent: Any, child: Any) -> None:
        self._thread_safe_cbk(self._add_edge, parent=parent, child=child)

    def _add_edge(self, parent: Any, child: Any) -> None:
        self.db.add_edge(parent, child)
        self.save()

    def remove_edge(self, parent: Any, child: Any) -> None:
        self._thread_safe_cbk(self._remove_edge, parent=parent, child=child)

    def _remove_edge(self, parent: Any, child: Any) -> None:
        self.db.remove_edge(parent, child)
        self.save()

    def visualize(self, seed: int = 3113794652, figsize: tuple = (20, 10)) -> None:
        plt.figure(figsize=figsize)
        pos = nx.spring_layout(self.db, seed=seed)
        return nx.draw_networkx(self.db, pos=pos, with_labels=True)

    def nodes(self) -> Iterable:
        return self.db.nodes(data=True)

    def edges(self) -> Iterable:
        return self.db.edges()

    def get_predecessors(self, uid: UID) -> list:
        return self.db.predecessors(uid)

    def get_successors(self, uid: UID) -> list:
        return self.db.successors(uid)

    def is_parent(self, parent: Any, child: Any) -> bool:
        parents = self.db.predecessors(child)
        return parent in parents

    def save(self) -> None:
        bytes = _serialize(self.db, to_bytes=True)
        with open(self.path_str, "wb") as f:
            f.write(bytes)

    def _filter_nodes_by(self, uid: UID, qks: QueryKeys) -> bool:
        node_data = self.db.nodes[uid]["data"]
        matches = []
        for qk in qks.all:
            matches.append(getattr(node_data, qk.key) == qk.value)
        # AND matches
        return all(matches)

    def subgraph(self, qks: QueryKeys) -> Any:
        filter_func = partial(self._filter_nodes_by, qks=qks)
        return nx.subgraph_view(self.db, filter_node=filter_func)

    def topological_sort(self, subgraph: Any) -> Any:
        return list(nx.topological_sort(subgraph))

    @staticmethod
    def _load_from_path(file_path: str) -> None:
        with open(file_path, "rb") as f:
            bytes = f.read()
        return _deserialize(blob=bytes, from_bytes=True)


@serializable()
class InMemoryGraphConfig(StoreConfig):
    __canonical_name__ = "InMemoryGraphConfig"

    store_type: type[BaseGraphStore] = NetworkXBackingStore
    client_config: StoreClientConfig = Field(default_factory=InMemoryStoreClientConfig)
    locking_config: LockingConfig = Field(default_factory=ThreadingLockingConfig)


@serializable()
class ActionGraphStore:
    pass


@serializable()
class InMemoryActionGraphStore(ActionGraphStore):
    __canonical_name__ = "InMemoryActionGraphStore"

    def __init__(self, store_config: StoreConfig, reset: bool = False):
        self.store_config: StoreConfig = store_config
        self.graph: BaseGraphStore = self.store_config.store_type(
            self.store_config, reset
        )

    def set(
        self,
        node: NodeActionData,
        credentials: SyftVerifyKey,
        parent_uids: list[UID] | None = None,
    ) -> Result[NodeActionData, str]:
        if self.graph.exists(uid=node.id):
            return Err(f"Node already exists in the graph: {node}")

        self.graph.set(uid=node.id, data=node)

        if parent_uids is None:
            parent_uids = []

        for parent_uid in parent_uids:
            result = self.add_edge(
                parent=parent_uid,
                child=node.id,
                credentials=credentials,
            )
            if result.is_err():
                return result

        return Ok(node)

    def get(
        self,
        uid: UID,
        credentials: SyftVerifyKey,
    ) -> Result[NodeActionData, str]:
        # 🟡 TODO: Add permission check
        if self.graph.exists(uid=uid):
            node_data = self.graph.get(uid=uid)
            return Ok(node_data)
        return Err(f"Node does not exists with id: {uid}")

    def delete(
        self,
        uid: UID,
        credentials: SyftVerifyKey,
    ) -> Result[bool, str]:
        # 🟡 TODO: Add permission checks
        if self.graph.exists(uid=uid):
            self.graph.delete(uid=uid)
            return Ok(True)
        return Err(f"Node does not exists with id: {uid}")

    def update(
        self,
        uid: UID,
        data: NodeActionDataUpdate,
        credentials: SyftVerifyKey,
    ) -> Result[NodeActionData, str]:
        # 🟡 TODO: Add permission checks
        node_data = self.graph.get(uid=uid)
        if node_data is not None:
            for key, val in data.to_dict(exclude_empty=True).items():
                setattr(node_data, key, val)
            self.graph.update(uid=uid, data=node_data)
            return Ok(node_data)
        return Err(f"Node does not exists for uid: {uid}")

    def update_non_mutated_successor(
        self,
        node_id: UID,
        nm_successor_id: UID,
        credentials: SyftVerifyKey,
    ) -> Result[NodeActionData, str]:
        """
        Used when a node is a mutagen and to update non-mutated
        successor for all nodes between node_id and nm_successor_id
        """
        node_data = self.graph.get(uid=node_id)

        data = NodeActionDataUpdate(
            next_mutagen_node=nm_successor_id,
            last_nm_mutagen_node=nm_successor_id,
            is_mutated=True,
        )

        if not node_data.is_mutated:
            # If current node is not mutated, then mark it as mutated
            return self.update(uid=node_id, data=data, credentials=credentials)
        else:
            # loop through successive mutagen nodes and
            # update their last_nm_mutagen_node id
            while node_id != nm_successor_id:
                node_data = self.graph.get(uid=node_id)

                # If node is the last added mutagen node,
                # then in that case its `next_mutagen_node` will be None
                # Therefore update its values to nm_successor_id
                next_mutagen_node = (
                    nm_successor_id
                    if node_data.next_mutagen_node is None
                    else node_data.next_mutagen_node
                )

                data = NodeActionDataUpdate(
                    last_nm_mutagen_node=nm_successor_id,
                    is_mutated=True,
                    next_mutagen_node=next_mutagen_node,
                )

                # Update each successive mutagen node
                result = self.update(
                    uid=node_id,
                    data=data,
                    credentials=credentials,
                )
                node_id = node_data.next_mutagen_node

            return result

    def _get_last_non_mutated_mutagen(
        self, credentials: SyftVerifyKey, uid: UID
    ) -> Result[UID, str]:
        node_data = self.graph.get(uid=uid)
        if node_data.is_mutated:
            return Ok(node_data.last_nm_mutagen_node)

        return Ok(uid)

    def add_edge(
        self,
        parent: UID,
        child: UID,
        credentials: SyftVerifyKey,
    ) -> Result[bool, str]:
        if not self.graph.exists(parent):
            return Err(f"Node does not exists for uid (parent): {parent}")

        if not self.graph.exists(child):
            return Err(f"Node does not exists for uid (child): {child}")

        result = self._get_last_non_mutated_mutagen(
            uid=parent,
            credentials=credentials,
        )

        if result.is_err():
            return result

        new_parent = result.ok()

        self.graph.add_edge(parent=new_parent, child=child)

        return Ok(True)

    def is_parent(self, parent: UID, child: UID) -> Result[bool, str]:
        if self.graph.exists(child):
            parents = self.graph.get_predecessors(child)
            result = parent in parents
            return Ok(result)
        return Err(f"Node doesn't exists for id: {child}")

    def query(
        self,
        qks: QueryKey | QueryKeys,
        credentials: SyftVerifyKey,
    ) -> Result[list[NodeActionData], str]:
        if isinstance(qks, QueryKey):
            qks = QueryKeys(qks=[qks])
        subgraph = self.graph.subgraph(qks=qks)
        return Ok(self.graph.topological_sort(subgraph=subgraph))

    def nodes(self, credentials: SyftVerifyKey) -> Result[list, str]:
        return Ok(self.graph.nodes())

    def edges(self, credentials: SyftVerifyKey) -> Result[list, str]:
        return Ok(self.graph.edges())
