# Enclave Development

## Building Attestion Containers

NOTE: Even on Arm machines, we build x64 images.
As some dependent packages in the dockerfile do not have arm64 equivalent.
It would take 10 minutes to build the image in emulation for the first time
in Arm machines.After which , the subsequent builds would be instant.

```sh
cd packages/grid/enclave/attestation && \
docker build -f attestation.dockerfile  . -t attestation:0.1 --platform linux/amd64
```

## Running the container in development mode

```sh
cd packages/grid/enclave/attestation && \
docker run -it --rm -e DEV_MODE=True -p 4455:4455 -v $(pwd)/server:/app/server attestation:0.1
```

## For fetching attestation report by FastAPI

### CPU Attestation

```sh
docker run -it --rm --privileged \
  -p 4455:4455 \
  -v /sys/kernel/security:/sys/kernel/security \
  -v /dev/tpmrm0:/dev/tpmrm0 attestation:0.1
```

```sh
curl localhost:4455/attest/cpu
```

### GPU Attestation

#### Nvidia GPU Requirements

We would need to install Nvidia Container Toolkit on host system and ensure we have CUDA Drivers installed.
Link: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html

```sh
docker run -it --rm --privileged --gpus all --runtime=nvidia \
  -p 4455:4455 \
  -v /sys/kernel/security:/sys/kernel/security \
  -v /dev/tpmrm0:/dev/tpmrm0 attestation:0.1
```

```sh
curl localhost:4455/attest/gpu
```

## For fetching attestation report directly by docker

### CPU Attestation

```sh
docker run -it --rm --privileged \
  -v /sys/kernel/security:/sys/kernel/security \
  -v /dev/tpmrm0:/dev/tpmrm0 attestation:0.1 /bin/bash
```

In the shell run

```sh
./AttestationClient
```

This would return either True or False indicating status of attestation

This could also be customized with Appraisal Policy

To retrieve JWT from Microsoft Azure Attestation (MAA)

```sh
./AttestationClient -o token
```

### For GPU Attestation

```sh
docker run -it --rm --privileged --gpus all --runtime=nvidia \
  -v /sys/kernel/security:/sys/kernel/security \
  -v /dev/tpmrm0:/dev/tpmrm0 attestation:0.1 /bin/bash
```

Invoke python shell
In the python shell run

```python3
from nv_attestation_sdk import attestation


NRAS_URL="https://nras.attestation.nvidia.com/v1/attest/gpu"
client = attestation.Attestation()
client.set_name("thisServer1")
client.set_nonce("931d8dd0add203ac3d8b4fbde75e115278eefcdceac5b87671a748f32364dfcb")
print ("[RemoteGPUTest] server name :", client.get_name())

client.add_verifier(attestation.Devices.GPU, attestation.Environment.REMOTE, NRAS_URL, "")
client.attest()
```

### Instructions for Development (Devspace)

We could launch an enclave stack by the command.

```sh
tox -e dev.k8s.launch.enclave
```

### Local Client-side Verification

Use the following function to perform local, client-side verification of tokens. They expire quick.

```python3
def verify_token(token: str, token_type: str):
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
        type (str): Type of the token which dictates the verification process; expected values are 'cpu' or 'gpu'.

    Returns:
        bool: True if the JWT is successfully verified, False otherwise.

    Raises:
        Exception: Raises various exceptions internally but catches them to return False, except for
                   printing error messages related to the specific failures (e.g., key not found, invalid certificate).

    Example usage:
        verify_token('your.jwt.token', 'cpu')
        verify_token('your.jwt.token', 'gpu')

    Note:
        - The function prints out details about the verification process and errors, if any.
        - Ensure that the cryptography and PyJWT libraries are properly installed and updated in your environment.
    """
    import jwt
    import json
    import base64
    import requests
    from jwt.algorithms import RSAAlgorithm
    from cryptography.x509 import load_der_x509_certificate
    from cryptography.hazmat.primitives import serialization


    # Determine JWKS URL based on the token type
    if token_type.lower() == "gpu":
        jwks_url = 'https://nras.attestation.nvidia.com/.well-known/jwks.json'
    else:
        unverified_header = jwt.get_unverified_header(token)
        jwks_url = unverified_header['jku']

    # Fetch the JWKS from the endpoint
    jwks = requests.get(jwks_url).json()

    # Get the key ID from the JWT header
    header = jwt.get_unverified_header(token)
    kid = header['kid']

    # Find the key with the matching kid in the JWKS
    key = next((item for item in jwks["keys"] if item["kid"] == kid), None)
    if not key:
        print("Public key not found in JWKS list.")
        return False

    # Convert the key based on the token type
    if token_type.lower() == "gpu" and "x5c" in key:
        try:
            cert_bytes = base64.b64decode(key['x5c'][0])
            cert = load_der_x509_certificate(cert_bytes)
            public_key = cert.public_key()
        except Exception as e:
            print("Failed to process certificate:", str(e))
            return False
    elif token_type.lower() == "cpu":
        try:
            public_key = RSAAlgorithm.from_jwk(key)
        except Exception as e:
            print("Failed to convert JWK to PEM:", str(e))
            return False
    else:
        print("Invalid token_type or key information.")
        return False

    # Verify the JWT using the public key
    try:
        payload = jwt.decode(token, public_key, algorithms=[header['alg']], options={"verify_exp": True})
        print("JWT Payload:", json.dumps(payload, indent=2))
        return True
    except jwt.ExpiredSignatureError:
        print("JWT token has expired.")
    except jwt.InvalidTokenError as e:
        print("JWT token signature is invalid:", str(e))

    return False
```
