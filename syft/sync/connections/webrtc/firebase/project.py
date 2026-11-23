"""The URLs of a Firebase project, or of the Firebase emulators."""

from dataclasses import dataclass

from syft.sync.connections.webrtc.webrtc_config import WebRTCConfig

AUTH_URL = "https://identitytoolkit.googleapis.com/v1"
TOKEN_URL = "https://securetoken.googleapis.com/v1"
FIRESTORE_URL = "https://firestore.googleapis.com/v1"
# The region and the name of the function in firebase/functions/main.py.
TURN_FUNCTION_URL = (
    "https://us-central1-{project_id}.cloudfunctions.net/turn_credentials"
)
# The emulator ports in firebase/firebase.json.
AUTH_EMULATOR_PORT = 9099
FIRESTORE_EMULATOR_PORT = 8080


@dataclass(frozen=True)
class FirebaseProject:
    project_id: str
    # The web API key of the project. It is not a secret.
    api_key: str
    # The REST API of Firebase Auth.
    auth_url: str
    # The REST API that gives new ID tokens.
    token_url: str
    # The REST API of Firestore.
    firestore_url: str
    # None on the emulators, because they run no Cloud Function.
    turn_function_url: str | None

    @classmethod
    def from_config(cls, config: WebRTCConfig) -> "FirebaseProject":
        """Return the real project of ``config``, or the emulators that replace
        it."""
        if config.firebase_emulator_host is None:
            return cls(
                project_id=config.firebase_project_id,
                api_key=config.firebase_api_key,
                auth_url=AUTH_URL,
                token_url=TOKEN_URL,
                firestore_url=FIRESTORE_URL,
                turn_function_url=TURN_FUNCTION_URL.format(
                    project_id=config.firebase_project_id
                ),
            )
        auth_emulator = f"http://{config.firebase_emulator_host}:{AUTH_EMULATOR_PORT}"
        return cls(
            project_id=config.firebase_project_id,
            api_key=config.firebase_api_key,
            auth_url=f"{auth_emulator}/identitytoolkit.googleapis.com/v1",
            token_url=f"{auth_emulator}/securetoken.googleapis.com/v1",
            firestore_url=(
                f"http://{config.firebase_emulator_host}:{FIRESTORE_EMULATOR_PORT}/v1"
            ),
            turn_function_url=None,
        )

    @property
    def documents_name(self) -> str:
        """The resource name of the root of the Firestore documents."""
        return f"projects/{self.project_id}/databases/(default)/documents"
