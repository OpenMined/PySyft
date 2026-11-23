# Firebase project for syft's WebRTC mode

In WebRTC mode, syft clients send their messages peer to peer. To find each
other, they use a Firebase project that your org sets up once:

- Firestore holds the signaling, the presence of each client, the peer
  requests, the key bundles and the peer info. `firestore.rules` controls who
  reads and writes each document.
- Firebase Authentication proves the email of each user, with a sign-in link
  that it sends to the email. A user needs no Google account.
- The Cloud Function `turn_credentials` gives signed-in users short-lived
  TURN credentials from Cloudflare. TURN relays the traffic of two peers that
  cannot connect directly.

No server or VM of your own takes part.

## Set up the project

You need Node.js, the Firebase CLI (`npm install -g firebase-tools`) and
Python 3.12.

1. In the [Firebase console](https://console.firebase.google.com), create a
   project. Upgrade it to the Blaze plan, because Cloud Functions need it.
2. Create a Firestore database in Native mode.
3. In Authentication, open "Sign-in method". Turn on "Email/Password", and
   in it "Email link (passwordless sign-in)".
4. In the Cloudflare dashboard, create a Realtime TURN key. Keep its token ID
   and its API token.
5. In this folder, install the function's dependencies, store the Cloudflare
   values as secrets, and deploy:

   ```bash
   python3.12 -m venv functions/venv
   functions/venv/bin/pip install -r functions/requirements.txt
   firebase login
   firebase functions:secrets:set CLOUDFLARE_TURN_KEY_ID --project <project-id>
   firebase functions:secrets:set CLOUDFLARE_TURN_KEY_API_TOKEN --project <project-id>
   firebase deploy --only firestore:rules,functions --project <project-id>
   ```

6. Give each user the project ID and the web API key. Both are in the project
   settings, on the "General" tab. The web API key is not a secret. If the
   field is empty, add a web app to the project.

## Log in as a user

```bash
export SYFT_FIREBASE_PROJECT_ID=<project-id>
export SYFT_FIREBASE_API_KEY=<web-api-key>
```

```python
import syft as sy

client = sy.login_do(email, webrtc=True)
```

The first login on a machine sends a sign-in link to the email. Paste the
link into the notebook. syft saves the session, so the next login asks
nothing.

## Run on the emulators

For development and tests, the Firebase emulators replace the project:

```bash
npx firebase-tools@15.35.0 emulators:start --only auth,firestore --project demo-syft
```

```python
from syft.sync.connections.webrtc.webrtc_config import WebRTCConfig

webrtc = WebRTCConfig(
    firebase_project_id="demo-syft",
    firebase_api_key="any-key",
    firebase_emulator_host="127.0.0.1",
)
client = sy.login_do(email, webrtc=webrtc)
```

The emulators run no Cloud Function, so clients get no ICE servers. Peers on
one machine need none. The emulators need Java 11 or later.
