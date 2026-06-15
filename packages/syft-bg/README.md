# syft-bg

Background services for SyftBox: email notifications and auto-approval for peers and jobs.

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
syft-bg auto-approve           # Create auto-approval object
syft-bg remove-auto-approval   # Remove files from an auto-approval object
syft-bg remove-peer            # Remove a peer from config
syft-bg list-auto-approvals    # List auto-approval objects
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

## Auto-Approval

Data owners can configure auto-approval objects that automatically approve matching jobs.
Each object specifies files to match by content (name + SHA256 hash) and optionally files to match by name only.

### Creating auto-approval objects

```bash
# Approve files for specific peers
syft-bg auto-approve main.py -p alice@uni.edu -p bob@co.com

# Approve multiple files with a name
syft-bg auto-approve main.py utils.py -n my_analysis

# Approve all files in a directory, allow params.json by name only
syft-bg auto-approve ./src/ -p alice@uni.edu -f params.json

# Use a base directory for relative path resolution
syft-bg auto-approve main.py -b ./project/ -f config.yaml
```

### Managing auto-approvals and peers

```bash
# List all auto-approval objects
syft-bg list-auto-approvals

# List a specific auto-approval object
syft-bg list-auto-approvals -n my_analysis

# Remove files from an auto-approval object
syft-bg remove-auto-approval utils.py -n my_analysis

# Remove a peer entirely
syft-bg remove-peer alice@uni.edu
```

### How validation works

When a job is submitted, the approval service checks:

1. Every file in the job must match an approved file entry
2. Content-matched files must have a matching SHA256 hash
3. If peers are specified, the submitter must be in the list

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

Auto-approves peers and jobs based on your config:

- **Peers**: Auto-accept connection requests from approved domains
- **Jobs**: Auto-approve if every submitted script matches an approved name + hash for that peer

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
