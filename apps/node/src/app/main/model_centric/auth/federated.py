# PyGrid imports
import base64
import json
import logging
import uuid

# Generic imports
import jwt
import requests

from ...core.codes import CYCLE, MODEL_CENTRIC_FL_EVENTS, MSG_FIELD, RESPONSE_MSG
from ..processes import process_manager


def verify_token(auth_token, model_name, model_version=None):

    kwargs = {"name": model_name}
    if model_version:
        kwargs["version"] = model_version
    server_config, _ = process_manager.get_configs(**kwargs)

    auth_config = server_config.get("authentication", {})
    endpoint = auth_config.get("endpoint", None)
    pub_key = auth_config.get("pub_key", None)
    secret = auth_config.get("secret", None)

    auth_enabled = endpoint or pub_key or secret

    if auth_enabled:
        if auth_token is None:
            return {
                "error": "Authentication is required, please pass an 'auth_token'.",
                "status": RESPONSE_MSG.ERROR,
            }
        else:
            payload = None
            error = False

            # Validate `auth_key` with passphrase (HMAC)
            if secret is not None:
                try:
                    payload = jwt.decode(auth_token, secret)
                except Exception as e:
                    logging.warning("Token validation against secret failed: " + str(e))
                    error = True

            # Validate `auth_token` with public key (RSA)
            if payload is None and pub_key is not None:
                try:
                    payload = jwt.decode(auth_token, pub_key)
                    error = False
                except Exception as e:
                    logging.warning(
                        "Token validation against public key failed: " + str(e)
                    )
                    error = True

            if error:
                return {
                    "error": "The 'auth_token' you sent is invalid.",
                    "status": RESPONSE_MSG.ERROR,
                }

            if endpoint is not None:
                external_api_verify_data = {"auth_token": f"{auth_token}"}
                verification_result = requests.post(
                    endpoint, data=json.dumps(external_api_verify_data)
                )

                if verification_result.status_code != 200:
                    return {
                        "error": "The 'auth_token' you sent did not pass 3rd party validation.",
                        "status": RESPONSE_MSG.ERROR,
                    }

            return {"status": RESPONSE_MSG.SUCCESS}
    else:
        # Authentication is not configured
        return {"status": RESPONSE_MSG.SUCCESS}
