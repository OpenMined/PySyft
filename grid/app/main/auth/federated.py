# PyGrid imports
from ..codes import MSG_FIELD, RESPONSE_MSG, CYCLE, FL_EVENTS
from ..processes import process_manager

# Generic imports
import jwt
import uuid
import json
import base64
import requests


def verify_token(auth_token, model_name):
    server_config, _ = process_manager.get_configs(name=model_name)
    auth_config = server_config.get("authentication", {})

    """stub DB vars"""
    JWT_VERIFY_API = auth_config.get("endpoint", None)
    RSA = auth_config.get("JWT_with_RSA", None)

    if RSA:
        pub_key = auth_config.get("pub_key", None)
    else:
        SECRET = auth_config.get("secret", "very long a$$ very secret key phrase")
    """end stub DB vars"""

    HIGH_SECURITY_RISK_NO_AUTH_FLOW = False if JWT_VERIFY_API is not None else True

    if not HIGH_SECURITY_RISK_NO_AUTH_FLOW:
        if auth_token is None:
            return {
                "error": "Authentication is required, please pass an 'auth_token'.",
                "status": RESPONSE_MSG.ERROR,
            }
        else:
            base64Header, base64Payload, signature = auth_token.split(".")
            header_str = base64.b64decode(base64Header)
            header = json.loads(header_str)
            _algorithm = header["alg"]

            if not RSA:
                payload_str = base64.b64decode(base64Payload)
                payload = json.loads(payload_str)
                expected_token = jwt.encode(
                    payload, SECRET, algorithm=_algorithm
                ).decode("utf-8")

                if expected_token != auth_token:
                    return {
                        "error": "The 'auth_token' you sent is invalid.",
                        "status": RESPONSE_MSG.ERROR,
                    }
            else:
                # we should check if RSA is true there is a pubkey string included during call to `host_federated_training`
                # here we assume it exists / no redundant check
                try:
                    jwt.decode(auth_token, pub_key, _algorithm)

                except Exception as e:
                    if e.__class__.__name__ == "InvalidSignatureError":
                        return {
                            "error": "The 'auth_token' you sent is invalid. " + str(e),
                            "status": RESPONSE_MSG.ERROR,
                        }

    external_api_verify_data = {"auth_token": f"{auth_token}"}
    verification_result = requests.get(
        "http://google.com"
    )  # test with get and google for now. using .post should result in failure
    # TODO:@MADDIE replace after we have a api to test with `verification_result = requests.post(JWT_VERIFY_API, data = json.dumps(external_api_verify_data))`

    if verification_result.status_code == 200:
        return {
            "auth_token": f"{auth_token}",
            "status": RESPONSE_MSG.SUCCESS,
        }
    else:
        return {
            "error": "The 'auth_token' you sent did not pass 3rd party verificaiton. ",
            "status": RESPONSE_MSG.ERROR,
        }
