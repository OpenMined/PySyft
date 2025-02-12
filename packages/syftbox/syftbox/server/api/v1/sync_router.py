import base64
import hashlib
import sqlite3
from collections import defaultdict
from typing import Iterator, List

import msgpack
import py_fast_rsync
from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from loguru import logger
from typing_extensions import Generator

from syftbox.lib.permissions import PermissionType
from syftbox.server.analytics import log_file_change_event
from syftbox.server.db.db import get_all_datasites
from syftbox.server.db.file_store import FileStore
from syftbox.server.db.schema import get_db
from syftbox.server.settings import ServerSettings, get_server_settings
from syftbox.server.users.auth import get_current_user

from ...models.sync_models import (
    ApplyDiffRequest,
    ApplyDiffResponse,
    BatchFileRequest,
    DiffRequest,
    DiffResponse,
    FileMetadata,
    FileMetadataRequest,
    FileRequest,
    RelativePath,
)


def get_db_connection(request: Request) -> Generator[sqlite3.Connection, None, None]:
    conn = get_db(request.state.server_settings.file_db_path)
    yield conn
    conn.close()


def get_file_store(request: Request) -> Generator[FileStore, None, None]:
    store = FileStore(
        server_settings=request.state.server_settings,
    )
    yield store


router = APIRouter(prefix="/sync", tags=["sync"])


@router.post("/get_diff", response_model=DiffResponse)
def get_diff(
    req: DiffRequest,
    file_store: FileStore = Depends(get_file_store),
    email: str = Depends(get_current_user),
) -> DiffResponse:
    try:
        file = file_store.get(req.path, email)
    except ValueError:
        raise HTTPException(status_code=404, detail="file not found")
    diff = py_fast_rsync.diff(req.signature_bytes, file.data)
    diff_bytes = base64.b85encode(diff).decode("utf-8")
    return DiffResponse(
        path=file.metadata.path.as_posix(),
        diff=diff_bytes,
        hash=file.metadata.hash,
    )


@router.post("/datasite_states", response_model=dict[str, list[FileMetadata]])
def get_datasite_states(
    file_store: FileStore = Depends(get_file_store),
    email: str = Depends(get_current_user),
) -> dict[str, list[FileMetadata]]:
    file_metadata = file_store.list_for_user(email=email)

    datasite_states = defaultdict(list)
    for metadata in file_metadata:
        user_email = metadata.path.parts[0]
        datasite_states[user_email].append(metadata)

    return dict(datasite_states)


@router.post("/dir_state", response_model=list[FileMetadata])
def dir_state(
    dir: RelativePath,
    file_store: FileStore = Depends(get_file_store),
    server_settings: ServerSettings = Depends(get_server_settings),
    email: str = Depends(get_current_user),
) -> list[FileMetadata]:
    return file_store.list_for_user(email=email, path=dir)


@router.post("/get_metadata", response_model=FileMetadata)
def get_metadata(
    req: FileMetadataRequest,
    file_store: FileStore = Depends(get_file_store),
    email: str = Depends(get_current_user),
) -> FileMetadata:
    try:
        metadata = file_store.get_metadata(req.path, email)
        return metadata
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/apply_diff", response_model=ApplyDiffResponse)
def apply_diffs(
    req: ApplyDiffRequest,
    file_store: FileStore = Depends(get_file_store),
    email: str = Depends(get_current_user),
) -> ApplyDiffResponse:
    try:
        file = file_store.get(req.path, email)
    except ValueError:
        raise HTTPException(status_code=404, detail="file not found")

    result = py_fast_rsync.apply(file.data, req.diff_bytes)
    new_hash = hashlib.sha256(result).hexdigest()

    if new_hash != req.expected_hash:
        raise HTTPException(status_code=400, detail="hash mismatch, skipped writing")

    file_store.put(req.path, result, user=email, check_permission=PermissionType.WRITE)

    log_file_change_event(
        "/sync/apply_diff",
        email=email,
        relative_path=req.path,
        file_store=file_store,
    )

    return ApplyDiffResponse(path=req.path, current_hash=new_hash, previous_hash=file.metadata.hash)


@router.post("/delete", response_class=JSONResponse)
def delete_file(
    req: FileRequest,
    file_store: FileStore = Depends(get_file_store),
    email: str = Depends(get_current_user),
) -> JSONResponse:
    log_file_change_event(
        "/sync/delete",
        email=email,
        relative_path=req.path,
        file_store=file_store,
    )

    file_store.delete(req.path, email)
    return JSONResponse(content={"status": "success"})


@router.post("/create", response_class=JSONResponse)
def create_file(
    file: UploadFile,
    file_store: FileStore = Depends(get_file_store),
    email: str = Depends(get_current_user),
) -> JSONResponse:
    relative_path = RelativePath(file.filename)
    if "%" in file.filename:
        raise HTTPException(status_code=400, detail="filename cannot contain '%'")

    if file_store.exists(relative_path):
        raise HTTPException(status_code=400, detail="file already exists")

    contents = file.file.read()

    file_store.put(
        relative_path,
        contents,
        user=email,
        check_permission=PermissionType.CREATE,
    )

    log_file_change_event(
        "/sync/create",
        email=email,
        relative_path=relative_path,
        file_store=file_store,
    )
    return JSONResponse(content={"status": "success"})


@router.post("/download", response_class=FileResponse)
def download_file(
    req: FileRequest,
    file_store: FileStore = Depends(get_file_store),
    email: str = Depends(get_current_user),
) -> FileResponse:
    try:
        abs_path = file_store.get(req.path, email).absolute_path
        return FileResponse(abs_path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/datasites", response_model=list[str])
def get_datasites(
    conn: sqlite3.Connection = Depends(get_db_connection),
    email: str = Depends(get_current_user),
) -> list[str]:
    return get_all_datasites(conn)


def file_streamer(files: List[RelativePath], file_store: FileStore, email: str) -> Iterator[bytes]:
    for path in files:
        try:
            file = file_store.get(path, email)
            metadata = {
                "path": file.metadata.path.as_posix(),
                "content": file.data,
            }
            yield msgpack.packb(metadata)
        except ValueError:
            logger.warning(f"File not found: {path}")
            continue


@router.post("/download_bulk")
def get_files(
    req: BatchFileRequest,
    file_store: FileStore = Depends(get_file_store),
    email: str = Depends(get_current_user),
) -> StreamingResponse:
    return StreamingResponse(file_streamer(req.paths, file_store, email), media_type="application/x-ndjson")
