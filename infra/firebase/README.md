# Firebase project for syft's WebRTC mode

In WebRTC mode, syft clients send their messages peer to peer. To find each
other, they use a Firebase project:

- The Realtime Database holds the peer requests, the list of peers that each
  user accepts (`acl/`), the key bundles, the peer info, and the signals: a
  hello when a client comes online, and the offers and answers of WebRTC.
  Each client reads its own signals as one stream. `database.rules.json`
  controls who reads and writes each path: a client can send signals only to
  a user who accepted it.
- Firebase Authentication proves the email of each user, with a sign-in link
  that it sends to the email. A user needs no Google account.
- The Cloud Function `turn_credentials` gives signed-in users TURN
  credentials from Cloudflare that work for 4 hours. TURN relays the traffic
  of two peers that cannot connect directly.

No server or VM of your own takes part.

OpenMined runs a project, and gives its project ID and web API key to its
users. An org that wants its own project sets it up as below.

## Set up a project

You need Node.js, the Firebase CLI (`npm install -g firebase-tools`), and
Python 3.12 with `uv`.

1. In the [Firebase console](https://console.firebase.google.com), create a
   project. Upgrade it to the Blaze plan, because Cloud Functions need it.
2. Create a Realtime Database in `us-central1`, in locked mode. The deploy
   below replaces its rules.
3. In Authentication, open "Sign-in method". Turn on "Email/Password", and
   in it "Email link (passwordless sign-in)". The console asks for more steps
   for apps; syft needs none of them.
4. In the Cloudflare dashboard, create a Realtime TURN key. Keep its token ID
   and its API token.
5. In this folder, install the function's dependencies, store the Cloudflare
   values as secrets, and deploy:

   ```bash
   uv venv functions/venv --python 3.12
   uv pip install --python functions/venv/bin/python -r functions/requirements.txt
   firebase login
   firebase functions:secrets:set CLOUDFLARE_TURN_KEY_ID --project <project-id>
   firebase functions:secrets:set CLOUDFLARE_TURN_KEY_API_TOKEN --project <project-id>
   firebase deploy --only database,functions --project <project-id> --force
   ```

   `--force` sets a cleanup policy that deletes old images of the function.

6. Give each user the project ID and the web API key, with your instructions
   and never in versioned source. Both are in the project settings, on the
   "General" tab. If the API key field is empty, add a web app to the
   project.

## Log in as a user

Set the project ID and the web API key that the project owner gave you,
before syft is imported:

```bash
export SYFT_FIREBASE_PROJECT_ID=<project-id>
export SYFT_FIREBASE_API_KEY=<web-api-key>
```

```python
import syft as sy

client = sy.login_do(email, webrtc=True)
```

The first login on a machine sends a sign-in link to the email. Copy the
link from the email, and paste it into the notebook. syft saves the session,
so the next login asks nothing.

## Run on the emulators

For development and tests, the Firebase emulators replace the project:

```bash
npx firebase-tools@15.35.0 emulators:start --only auth,database --project demo-syft
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
one machine need none. The emulators need Java 21 or later.
