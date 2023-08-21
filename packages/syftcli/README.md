# Syft CLI

## Development

```sh
pip install -e "packages/syftcli[dev]"
syftcli hello
```

## Building a Standalone Executable of Syft CLI

Keep the working directory as `packages/syftcli`

### Requirements

- Python >= 3.8

### Setup

It is recommended to create a fresh virtual environment for the build process.

### Installation

Inside the new virtual environment run

```sh
pip install -e ".[build]"
```

## Linux

```sh
pyinstaller --onefile src/syftcli/cli.py
```

## Windows

```sh
pyinstaller --onefile src\\syftcli\\cli.py
```

## Mac

To able to build a binary for you current platform (Intel or Apple Silicon) run

```sh
pyinstaller --onefile src/syftcli/cli.py
```

## Building universal binary for Mac

To be able to build a universal2 binary that could operate on Intel or Apple Silicon.

Initially install universal2 python: https://www.python.org/downloads/macos/

To check if you have universal2 python run the command

```sh
lipo -info `which python3`
```

This should return the output
`Architectures in the fat file: /Users/rasswanth/.venv/bin/python3 are: x86_64 arm64`

mentioning both arm and x86

To create a universal2 binary run

```sh
pyinstaller --onefile src/syftcli/cli.py --target-arch universal2
```

Finally, The binary executable could be found under
`/dist/cli`

#### Notes

To customize the binary name, you add `--name=<your_cli_name>` to the pyinstaller command
