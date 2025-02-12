from aiofiles import open as aopen
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse

from syftbox import __version__
from syftbox.client.routers.common import APIContext

router = APIRouter()


@router.get("/")
async def index() -> PlainTextResponse:
    return PlainTextResponse(f"SyftBox {__version__}")


@router.get("/version")
async def version() -> dict:
    return {"version": __version__}


@router.get("/logs")
async def get_logs(
    context: APIContext,
    limit: int = 256,
    offset: int = 0,
) -> JSONResponse:
    """Get last log lines from the log file.

    Args:
        limit: Maximum number of log lines to return
        offset: Number of lines to skip from the end of the log file
    """
    logs_dir = context.workspace.data_dir / "logs"

    try:
        log_files = sorted(logs_dir.glob("syftbox_*.log"), reverse=True)
        if not log_files:
            return JSONResponse(
                content={
                    "logs": [],
                    "total": 0,
                }
            )

        last_log_file = log_files[0]
        async with aopen(last_log_file, "r") as f:
            content = await f.readlines()

        total_logs = len(content)
        start_idx = max(total_logs - offset - limit, 0)
        end_idx = total_logs - offset if offset > 0 else total_logs

        return JSONResponse(
            content={
                "logs": content[start_idx:end_idx],
                "total": total_logs,
                "source": str(last_log_file),
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve logs: {str(e)}")


@router.get("/metadata")
async def metadata(ctx: APIContext) -> dict:
    return {"datasite": ctx.email}
