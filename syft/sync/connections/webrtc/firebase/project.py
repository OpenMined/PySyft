"""The URLs of a Firebase project, or of the Firebase emulators."""

from dataclasses import dataclass

from syft.sync.connections.webrtc.webrtc_config import WebRTCConfig

AUTH_URL = "https://identitytoolkit.googleapis.com/v1"
TOKEN_URL = "https://securetoken.googleapis.com/v1"
# The region and the name of the function in infra/firebase/functions/main.py.
TURN_FUNCTION_URL = (
    "https://us-central1-{project_id}.cloudfunctions.net/turn_credentials"
)
# The default Realtime Database of a project, in us-central1.
DATABASE_URL = "https://{database}.firebaseio.com"
DEFAULT_DATABASE = "{project_id}-default-rtdb"
# The emulator ports in infra/firebase/firebase.json.
AUTH_EMULATOR_PORT = 9099
DATABASE_EMULATOR_PORT = 9000


@dataclass(frozen=True)
class FirebaseProject:
    project_id: str
    # The web API key of the project. It is not a secret.
    api_key: str
    # The REST API of Firebase Auth.
    auth_url: str
    # The REST API that gives new ID tokens.
    token_url: str
    # The REST API of the Realtime Database.
    database_url: str
    # The database name that the emulator needs in each URL, or None for the
    # real project.
    database_namespace: str | None
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
                database_url=DATABASE_URL.format(
                    database=DEFAULT_DATABASE.format(
                        project_id=config.firebase_project_id
                    )
                ),
                database_namespace=None,
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
            database_url=(
                f"http://{config.firebase_emulator_host}:{DATABASE_EMULATOR_PORT}"
            ),
            database_namespace=DEFAULT_DATABASE.format(
                project_id=config.firebase_project_id
            ),
            turn_function_url=None,
        )
