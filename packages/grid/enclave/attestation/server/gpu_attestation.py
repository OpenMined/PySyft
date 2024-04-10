# third party
from loguru import logger
from nv_attestation_sdk import attestation

NRAS_URL = "https://nras.attestation.nvidia.com/v1/attest/gpu"


def attest_gpu() -> str:
    # Fetch report from Nvidia Attestation SDK
    client = attestation.Attestation("Attestation Node")

    # TODO: Add the ability to generate nonce later.
    logger.info("[RemoteGPUTest] node name : {}", client.get_name())

    client.add_verifier(
        attestation.Devices.GPU, attestation.Environment.REMOTE, NRAS_URL, ""
    )
    gpu_report = client.attest()
    logger.info("[RemoteGPUTest] report : {}, {}", gpu_report, type(gpu_report))

    return str(gpu_report)
