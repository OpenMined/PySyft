# stdlib
import base64
from enum import Enum

# third party
from cryptography.x509 import load_der_x509_certificate
import jwt
from jwt.algorithms import RSAAlgorithm
import requests
from result import Err
from result import Ok
from result import Result
from typing_extensions import Self


class AttestationType(str, Enum):
    # Define enum members with their corresponding JWKS URLs
    CPU = "CPU"
    GPU = "GPU"

    def __new__(cls, value: str) -> Self:
        JWKS_URL_MAP = {
            "CPU": "https://sharedeus2.eus2.attest.azure.net/certs",
            "GPU": "https://nras.attestation.nvidia.com/.well-known/jwks.json",
        }
        if value not in JWKS_URL_MAP:
            raise ValueError(f"JWKS URL not defined for token type: {value}")
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.jwks_url = JWKS_URL_MAP.get(value)
        return obj

    def __str__(self) -> str:
        return self.value


def verify_attestation_report(
    token: str, attestation_type: AttestationType = AttestationType.CPU
) -> Result[Ok[dict], Err[str]]:
    """
    Verifies a JSON Web Token (JWT) using a public key obtained from a JWKS (JSON Web Key Set) endpoint,
    based on the specified type of token ('cpu' or 'gpu'). The function handles two distinct processes
    for token verification depending on the type specified:

    - 'cpu': Fetches the JWKS from the 'jku' URL specified in the JWT's unverified header,
             finds the key by 'kid', and converts the JWK to a PEM format public key for verification.

    - 'gpu': Directly uses a fixed JWKS URL to retrieve the keys, finds the key by 'kid', and uses the
             'x5c' field to extract a certificate which is then used to verify the token.

    Parameters:
        token (str): The JWT that needs to be verified.
        token_type (AttestationType): The type of token to be verified (CPU or GPU).

    Returns:
        Result[Ok[dict], Err[str]]: A Result object containing the payload of the verified token if successful,
                                    or an Err object with an error message if the verification fails.
    """
    jwks_url = attestation_type.jwks_url
    unverified_header = jwt.get_unverified_header(token)

    try:
        # Fetch the JWKS from the endpoint
        jwks = requests.get(jwks_url).json()
    except Exception as e:
        return Err(f"Failed to fetch JWKS: {str(e)}")

    try:
        # Get the key ID from the JWT header and find the matching key in the JWKS
        kid = unverified_header["kid"]
        key = next((item for item in jwks["keys"] if item["kid"] == kid), None)
        if not key:
            return Err("Public key not found in JWKS list.")
    except Exception as e:
        return Err(f"Failed to process JWKS: {str(e)}")

    try:
        # Convert the key based on the token type
        if attestation_type == AttestationType.GPU and "x5c" in key:
            cert_bytes = base64.b64decode(key["x5c"][0])
            cert = load_der_x509_certificate(cert_bytes)
            public_key = cert.public_key()
        elif attestation_type == AttestationType.CPU:
            public_key = RSAAlgorithm.from_jwk(key)
        else:
            return Err("Invalid token type or key information.")
    except Exception as e:
        return Err(f"Failed to process public key: {str(e)}")

    try:
        # Verify the JWT using the public key
        payload = jwt.decode(
            token,
            public_key,
            algorithms=[unverified_header["alg"]],
            options={"verify_exp": True},
        )
        return Ok(payload)
    except jwt.ExpiredSignatureError:
        return Err("JWT token has expired.")
    except jwt.InvalidTokenError as e:
        return Err(f"JWT token signature is invalid: {str(e)}")
