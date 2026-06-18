"""Minimal FastAPI surface for the inference service.

The docker image mounts this router into the attestation app so everything
serves on the single Confidential Spaces port (8080).
"""

from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel

from enclave_model_api.service import InferenceService


class InferRequest(BaseModel):
    query: str
    max_new_tokens: int = 200


class InferResponse(BaseModel):
    completion: str
    stats: dict


def build_router(service: InferenceService) -> APIRouter:
    router = APIRouter()

    @router.post("/infer", response_model=InferResponse)
    def infer(request: InferRequest) -> InferResponse:
        if not service.loaded:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded yet — weights may still be syncing.",
            )
        completion, stats = service.infer(
            request.query, max_new_tokens=request.max_new_tokens
        )
        return InferResponse(completion=completion, stats=stats)

    @router.get("/model-status")
    def model_status() -> dict:
        return {
            "model_size": service.model_size,
            "weights_present": service.weights_present,
            "model_loaded": service.loaded,
        }

    return router


def create_app(service: InferenceService) -> FastAPI:
    """Standalone app for tests and local runs (docker combines with attestation)."""
    app = FastAPI(title="Syft Enclave Inference", version="0.1.0")
    app.include_router(build_router(service))

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    return app
