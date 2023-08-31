# Syft CLI

## Development

Install the syftcli package in editable mode

```sh
pip install -e "packages/syftcli[dev]"
syftcli hello
```

Run as a module

```sh
cd packages/syftcli
python -m syftcli.cli hello
```

Debug in VSCode with the following `launch.json`

```json
{
    "name": "Python: Syft CLI",
    "type": "python",
    "request": "launch",
    "module": "syftcli.cli",
    "args": ["bundle", "create"], # CLI command to run
    "console": "integratedTerminal",
    "cwd": "${workspaceFolder}/packages/syftcli",
    "justMyCode": true
}
```

## Building a standalone executable

### Requirements

- Python >= 3.8

### The fast way

```sh
tox -re syftcli.build
```

NOTE: The `-re` flag is neecessary to re-create tox environment for the build.

Once the build is successful, the executable will be at `packages/syft/dist`

### The manual way

Keep the working directory as `packages/syftcli`

### Setup

It is recommended to create a fresh virtual environment for the build process.

```sh
cd packages/syftcli

rm -rf .venv
python -m venv .venv

source .venv/bin/activate

pip install -e ".[build]"
```

### Linux/WSL

```sh
pyinstaller --onefile src/cli.py --name=syftcli
```

### Windows

```sh
pyinstaller --onefile src\\cli.py --name=syftcli
```

### Mac

To able to build a binary for you current platform (Intel or Apple Silicon) run

```sh
pyinstaller --onefile src/cli.py --name=syftcli
```

### Universal binary for Mac

Universal binary is a single executable that can run on both Intel and Apple Silicon.

Make sure you have Universal2 Python installed on your macOS: https://www.python.org/downloads/macos/

Check if you have universal2 python

```sh
lipo -info `which python3`
```

This should return the output mentioning both arm and x86

```
Architectures in the fat file: .venv/bin/python3 are: x86_64 arm64
```

Once confirmed, run the following to create a universal2 binary

```sh
pyinstaller --onefile src/cli.py --target-arch universal2 --name=syftcli
```

TThe binary executable could be found under `packages/syftcli/dist/syftcli`
