from pathlib import Path
from typing import List, Optional

import wcmatch.glob
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from jinja2 import Environment, FileSystemLoader

from syftbox.client.exceptions import SyftPluginException
from syftbox.client.plugins.sync.local_state import SyncStatusInfo
from syftbox.client.plugins.sync.manager import SyncManager
from syftbox.client.plugins.sync.types import SyncStatus
from syftbox.client.routers.common import APIContext

router = APIRouter()
jinja_env = Environment(loader=FileSystemLoader("syftbox/assets/templates"))


def get_sync_manager(context: APIContext) -> SyncManager:
    if context.plugins is None:
        raise HTTPException(status_code=500, detail="Plugin manager not initialized")
    try:
        return context.plugins.sync_manager
    except SyftPluginException as e:
        raise HTTPException(status_code=500, detail=f"Failed to get sync manager: {e}")


def _get_queued_items(sync_manager: SyncManager) -> List[SyncStatusInfo]:
    # make copy to avoid changing size during iteration
    queued_items = list(sync_manager.queue.all_items.values())
    return [
        SyncStatusInfo(
            path=item.data.path,
            status=SyncStatus.QUEUED,
            timestamp=item.enqueued_at,
        )
        for item in queued_items
    ]


def _get_items_from_localstate(sync_manager: SyncManager) -> List[SyncStatusInfo]:
    return list(sync_manager.local_state.status_info.values())


def get_all_status_info(sync_manager: SyncManager) -> List[SyncStatusInfo]:
    """
    Return all status info from both the queue and local state.
    NOTE: the result might contain duplicates if the same path is present in both.
    """
    queued_items = _get_queued_items(sync_manager)
    localstate_items = _get_items_from_localstate(sync_manager)
    return queued_items + localstate_items


def deduplicate_status_info(status_info_list: List[SyncStatusInfo]) -> List[SyncStatusInfo]:
    """Deduplicate status info by path, keeping the entry with latest timestamp"""
    path_to_info: dict[Path, SyncStatusInfo] = {}
    for info in status_info_list:
        existing_info = path_to_info.get(info.path)
        if not existing_info or info.timestamp > existing_info.timestamp:
            path_to_info[info.path] = info
    return list(path_to_info.values())


def sort_status_info(status_info_list: List[SyncStatusInfo], order_by: str, order: str) -> List[SyncStatusInfo]:
    if order_by.lower() not in SyncStatusInfo.model_fields:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid order_by field: {order_by}. Available fields: {list(SyncStatusInfo.model_fields.keys())}",
        )
    if order.lower() not in ["asc", "desc"]:
        raise HTTPException(status_code=400, detail=f"Invalid order: {order}, expected 'asc' or 'desc'")

    return list(
        sorted(
            status_info_list,
            key=lambda x: getattr(x, order_by),
            reverse=order.lower() == "desc",
        )
    )


def filter_by_path_glob(items: List[SyncStatusInfo], pattern: Optional[str]) -> List[SyncStatusInfo]:
    if not pattern:
        return items

    result = []
    for item in items:
        if wcmatch.glob.globmatch(item.path.as_posix(), pattern, flags=wcmatch.glob.GLOBSTAR):
            result.append(item)
    return result


def apply_limit_offset(items: List[SyncStatusInfo], limit: Optional[int], offset: int) -> List[SyncStatusInfo]:
    if offset:
        items = items[offset:]
    if limit:
        items = items[:limit]
    return items


@router.get("/health")
def health_check(sync_manager: SyncManager = Depends(get_sync_manager)) -> JSONResponse:
    if not sync_manager.is_alive():
        raise HTTPException(status_code=503, detail="Sync service unavailable")
    return JSONResponse(content={"status": "ok"})


@router.get("/state")
def get_status_info(
    order_by: str = "timestamp",
    order: str = "desc",
    path_glob: Optional[str] = None,
    sync_manager: SyncManager = Depends(get_sync_manager),
    limit: Optional[int] = None,
    offset: int = 0,
) -> List[SyncStatusInfo]:
    all_items = get_all_status_info(sync_manager)
    items_deduplicated = deduplicate_status_info(all_items)
    items_filtered = filter_by_path_glob(items_deduplicated, path_glob)
    items_sorted = sort_status_info(items_filtered, order_by, order)
    items_paginated = apply_limit_offset(items_sorted, limit, offset)
    return items_paginated


@router.get("/")
def sync_dashboard(context: APIContext) -> HTMLResponse:
    template = jinja_env.get_template("sync_dashboard.jinja2")
    return HTMLResponse(template.render(base_url=context.config.client_url))
