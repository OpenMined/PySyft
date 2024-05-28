# stdlib
import os
import sys

# third party
from fastapi import FastAPI
from loguru import logger

# relative
from .attestation_models import CPUAttestationResponseModel
from .attestation_models import GPUAttestationResponseModel
from .attestation_models import ResponseModel
from .cpu_attestation import attest_cpu
from .gpu_attestation import attest_gpu

# Logging Configuration
log_level = os.getenv("APP_LOG_LEVEL", "INFO").upper()
logger.remove()
logger.add(sys.stderr, colorize=True, level=log_level)

app = FastAPI(title="Attestation API")


@app.get("/", response_model=ResponseModel)
async def read_root() -> ResponseModel:
    return ResponseModel(message="Server is running")


@app.get("/attest/cpu", response_model=CPUAttestationResponseModel)
async def attest_cpu_endpoint() -> CPUAttestationResponseModel:
    cpu_attest_res, cpu_attest_token = attest_cpu()
    return CPUAttestationResponseModel(result=cpu_attest_res, token=cpu_attest_token)


@app.get("/attest/gpu", response_model=GPUAttestationResponseModel)
async def attest_gpu_endpoint() -> GPUAttestationResponseModel:
    gpu_attest_res, gpu_attest_token = attest_gpu()
    return GPUAttestationResponseModel(result=gpu_attest_res, token=gpu_attest_token)
