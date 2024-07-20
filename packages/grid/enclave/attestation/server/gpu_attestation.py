# stdlib
import io
import re
import sys

# third party
from loguru import logger
from nv_attestation_sdk import attestation

# relative
from .attestation_constants import NRAS_URL


# Function to process captured output to extract the token
def extract_token(captured_value: str) -> str:
    match = re.search(r"Entity Attestation Token is (\S+)", captured_value)
    if match:
        token = match.group(1)  # Extract the token, which is in group 1 of the match
        return token
    else:
        return "Token not found"


def attest_gpu() -> tuple[str, str]:
    # Fetch report from Nvidia Attestation SDK
    client = attestation.Attestation("Attestation Server")

    # TODO: Add the ability to generate nonce later.
    logger.info("[RemoteGPUTest] server name : {}", client.get_name())

    client.add_verifier(
        attestation.Devices.GPU, attestation.Environment.REMOTE, NRAS_URL, ""
    )

    # Step 1: Redirect stdout
    original_stdout = sys.stdout  # Save a reference to the original standard output
    captured_output = io.StringIO()  # Create a StringIO object to capture output
    sys.stdout = captured_output  # Redirect stdout to the StringIO object

    # Step 2: Call the function
    gpu_report = client.attest()

    # Step 3: Get the content of captured output and reset stdout
    captured_value = captured_output.getvalue()
    sys.stdout = original_stdout  # Reset stdout to its original state

    # Step 4: Extract the token from the captured output
    token = extract_token(captured_value)

    logger.info("[RemoteGPUTest] report : {}, {}", gpu_report, type(gpu_report))
    return str(gpu_report), token
