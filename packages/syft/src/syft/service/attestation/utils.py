# stdlib
import base64

# third party
from cryptography.x509 import load_der_x509_certificate
import jwt
from jwt.algorithms import RSAAlgorithm
import requests
from result import Err
from result import Ok
from result import Result


def verify_attestation_report(token: str) -> Result[Ok[dict], Err[str]]:
    """
    Verifies a JSON Web Token (JWT) using a public key obtained from a JWKS (JSON Web Key Set) endpoint.
    The function handles two distinct processes for token verification depending on the token type:

    - 'cpu': Fetches the JWKS from the 'jku' URL specified in the JWT's unverified header,
             finds the key by 'kid', and converts the JWK to a PEM format public key for verification.

    - 'gpu': Directly uses a fixed JWKS URL to retrieve the keys, finds the key by 'kid', and uses the
             'x5c' field to extract a certificate which is then used to verify the token.

    Parameters:
        token (str): The JWT that needs to be verified.

    Returns:
        Result[Ok[dict], Err[str]]: A Result object containing the payload of the verified token if successful,
                                    or an Err object with an error message if the verification fails.
    """
    try:
        # Attempt to retrieve 'jku' from the unverified header to determine the JWKS URL
        unverified_header = jwt.get_unverified_header(token)
        # TODO this is vulnerable. Hardcode the jwks_url to the actual Azure Attestation Service URL
        jwks_url = unverified_header.get("jku", "")
        token_type = "cpu" if jwks_url else "gpu"
        if not jwks_url:
            jwks_url = "https://nras.attestation.nvidia.com/.well-known/jwks.json"
    except Exception as e:
        return Err(f"Failed to determine JWKS URL: {str(e)}")

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
        if token_type == "gpu" and "x5c" in key:
            cert_bytes = base64.b64decode(key["x5c"][0])
            cert = load_der_x509_certificate(cert_bytes)
            public_key = cert.public_key()
        elif token_type == "cpu":
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
