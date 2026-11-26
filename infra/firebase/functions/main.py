"""The Cloud Function that gives syft's WebRTC clients TURN credentials."""

import json
import logging

from cloudflare_turn import CloudflareTurnConfig, generate_ice_servers
from firebase_admin import auth, initialize_app
from firebase_functions import https_fn
from firebase_functions.params import SecretParam

# The syft client builds the URL of the function from this region.
REGION = "us-central1"
# The offering client opens a new peer connection before this time ends.
CREDENTIALS_TTL_SECONDS = 4 * 3600
BEARER_PREFIX = "Bearer "
CLOUDFLARE_TURN_KEY_ID = SecretParam("CLOUDFLARE_TURN_KEY_ID")
CLOUDFLARE_TURN_KEY_API_TOKEN = SecretParam("CLOUDFLARE_TURN_KEY_API_TOKEN")

logger = logging.getLogger(__name__)
initialize_app()


@https_fn.on_request(
    region=REGION, secrets=[CLOUDFLARE_TURN_KEY_ID, CLOUDFLARE_TURN_KEY_API_TOKEN]
)
def turn_credentials(request: https_fn.Request) -> https_fn.Response:
    """Return ``{"ice_servers": [...], "lifetime_seconds": ...}`` with new TURN
    credentials.

    The request must carry the Firebase ID token of a user with a verified
    email, as ``Authorization: Bearer <token>``.
    """
    header = request.headers.get("Authorization", "")
    if not header.startswith(BEARER_PREFIX):
        return error_response(401, "Send a Firebase ID token as a Bearer token")
    try:
        token = auth.verify_id_token(header.removeprefix(BEARER_PREFIX))
    except (auth.InvalidIdTokenError, ValueError) as err:
        # The reason can describe the setup of the function, so only the log
        # of the function gets it.
        logger.warning("Refused a Firebase ID token: %r", err)
        return error_response(401, "The Firebase ID token is not valid")
    if token.get("email_verified") is not True:
        return error_response(403, "The email of the user is not verified")
    config = CloudflareTurnConfig(
        key_id=CLOUDFLARE_TURN_KEY_ID.value,
        api_token=CLOUDFLARE_TURN_KEY_API_TOKEN.value,
        ttl_seconds=CREDENTIALS_TTL_SECONDS,
    )
    body = json.dumps(
        {
            "ice_servers": generate_ice_servers(config),
            "lifetime_seconds": CREDENTIALS_TTL_SECONDS,
        }
    )
    return https_fn.Response(body, status=200, content_type="application/json")


def error_response(status: int, reason: str) -> https_fn.Response:
    body = json.dumps({"error": reason})
    return https_fn.Response(body, status=status, content_type="application/json")
