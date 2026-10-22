# syft-bg

Background services for SyftBox: email notifications, running api calls, and approving peers.

## Installation

```bash
pip install syft-bg
```

##

**[Python API docs](docs/python-api.md)** — for python api docs, go here instead.

## Quick Start

```bash
syft-bg init -e you@example.com          # Create config
syft-bg ensure-running notify approve    # Start services
syft-bg status                           # Check what's running
```

With custom settings:

```bash
syft-bg init -e user@example.com -r ~/SyftBox -t ~/token.json
syft-bg ensure-running notify approve
```

### Python API

For notebooks and scripts, see the [Python API docs](docs/python-api.md).

## Commands

```bash
syft-bg                        # TUI dashboard
syft-bg init -e <email>              # Create config
syft-bg ensure-running <services>    # Start services if not already running
syft-bg setup-status                 # Check environment (credentials, tokens, config)
syft-bg status                       # Show service status
syft-bg start [service]        # Start all or specific service
syft-bg stop [service]         # Stop all or specific service
syft-bg restart [service]      # Restart all or specific service
syft-bg logs <service>         # View logs (notify or approve)
syft-bg create-api             # Create an api
syft-bg remove-api             # Remove an api
syft-bg remove-peer            # Remove a peer from apis
syft-bg list-apis              # List apis
syft-bg install                # Install systemd service (auto-start on boot)
syft-bg uninstall              # Remove systemd service
```

## Starting and Stopping Services

```bash
syft-bg start               # Start all services
syft-bg start notify        # Start a specific service
syft-bg stop                # Stop all services
syft-bg stop approve        # Stop a specific service
syft-bg restart             # Restart all services
syft-bg restart notify      # Restart a specific service
```

Use `ensure-running` to start services only if they aren't already running:

```bash
syft-bg ensure-running notify approve
syft-bg ensure-running notify approve --restart  # Force restart
```

## Apis

An api is a job the data owner approved in advance, which peers call with their own
arguments: `client.api.<name>(...)`. A call sends only the arguments. The data owner's
client adds the api's code, approves the call, and the approve service runs it, with
the api's input and output policies. See [docs/jobs.md](../../docs/jobs.md).

### Creating an api

Prefer building it from a job you have reviewed, with `syft_bg.create_api_from_job(job)`.
From the command line, an api pins `run.sh` and the rest of its code by content,
and takes `config.yaml` and the files a call sends (such as the params JSON) by
name. Paths are relative to the job submission root.

```bash
cd ./job   # a submission holding code/main.py, code/params.json, run.sh and config.yaml

syft-bg create-api code/main.py run.sh -b . -f config.yaml -f code/params.json -p alice@uni.edu -n my_analysis
```

### Managing apis and peers

```bash
syft-bg list-apis                 # list all apis
syft-bg list-apis -n my_analysis  # show one api
syft-bg remove-api my_analysis    # remove an api
syft-bg remove-peer alice@uni.edu # remove a peer from every api
```

## CLI Flags for `syft-bg init`

| Flag                 | Description              |
| -------------------- | ------------------------ |
| `--email, -e`        | Data Owner email address |
| `--syftbox-root, -r` | SyftBox directory path   |
| `--token-path, -t`   | Path to OAuth token file |

## Environment Check

```bash
$ syft-bg setup-status

SYFT-BG ENVIRONMENT CHECK
==================================================

Checking credentials...
  ✓ credentials.json found at ~/.syft-bg/credentials.json

Checking authentication tokens...
  ✓ Gmail token: ~/.syft-bg/gmail_token.json
  ✓ Drive token: ~/.syft-bg/token_do.json

Checking configuration...
  ✓ Config file: ~/.syft-bg/config.yaml

--------------------------------------------------
✅ Environment ready! Run 'syft-bg start' to begin.
```

## OAuth Setup

Two OAuth tokens are required (same credentials.json, separate tokens):

1. **Gmail** → `gmail_token.json` (send email permission)
2. **Drive** → `drive_token.json` (read/write files permission)

To get credentials.json:

1. Go to Google Cloud Console → APIs & Services → Credentials
2. Create OAuth 2.0 Client ID (Desktop app)
3. Download as credentials.json
4. Place at `~/.syft-bg/credentials.json`

## Services

### notify

Sends email notifications via Gmail when:

- A peer requests to connect with you
- Your peer request is approved by someone
- A data scientist submits a job to you
- A job you submitted is approved
- A job completes (results ready)
- A job is rejected (with reason sent to the data scientist)

DO notifications are threaded per job (new → approved/rejected → completed in one Gmail conversation).

### approve

Runs api calls, and approves peers based on your config:

- **Peers**: Auto-accept connection requests from approved domains, and from the peers of your apis
- **Jobs**: Run every approved job; an api call arrives approved

## Configuration

Config stored at `~/.syft-bg/config.yaml` (Colab: `/content/drive/MyDrive/syft-creds/config.yaml`).

```yaml
do_email: you@example.com
syftbox_root: ~/SyftBox

notify:
  interval: 30
  monitor_jobs: true
  monitor_peers: true

approve:
  interval: 5
  jobs:
    enabled: true
    peers:
      alice@uni.edu:
        mode: strict
        scripts:
          - name: main.py
            hash: 'sha256:a1b2c3d4...'
          - name: utils.py
            hash: 'sha256:e5f6a7b8...'
      bob@co.com:
        mode: strict
        scripts:
          - name: main.py
            hash: 'sha256:c9d0e1f2...'
  peers:
    enabled: false
    approved_domains:
      - openmined.org
```

After editing, restart services:

```bash
syft-bg restart
```

## Systemd Integration

Auto-start syft-bg on boot (Linux):

```bash
syft-bg install    # Creates ~/.config/systemd/user/syft-bg.service
systemctl --user enable syft-bg
systemctl --user start syft-bg

# Check status
systemctl --user status syft-bg

# Remove
syft-bg uninstall
```

## Logs

```bash
syft-bg logs notify     # Notification service logs
syft-bg logs approve    # Approval service logs
syft-bg logs notify -f  # Follow logs in real-time
```

Log files stored at `~/.syft-bg/logs/`.

## Colab / Jupyter

See the [Python API docs](docs/python-api.md) for programmatic usage. Drive credentials are handled natively in Colab.

## Development

Run services in foreground for debugging:

```bash
syft-bg run-foreground --service notify   # Run notify in foreground
syft-bg run-foreground --service approve  # Run approve in foreground
syft-bg run-foreground --once             # Single check cycle, then exit
```
