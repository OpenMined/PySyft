# Syft Rust Core

![Rusty](https://images.unsplash.com/photo-1612590675174-3c497ca9c79a?auto=format&fit=crop&w=1280&q=80)

###### Photo by [Jacob Campbell](https://unsplash.com/@jacobsoup?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)

## _“An elegant weapon for a more civilized age.” — Obi-Wan Kenobi_

This repo is a coordinated effort to build a Rust Core from inside PySyft, with native
host language bindings. The result will be a Rust Crate called Syft with a Python
language wrapper called PySyft. The external API and user experience for Python users
should be transparent and they should never have to know the internals have changed.

The code here is heavily inspired by the original [syft_experimental](https://github.com/OpenMined/syft_experimental/tree/dev/syft) skunk works project.

## Mixed Structure

The folder structure looks like this:

```
OpenMined/PySyft
├── proto                                   <- Shared Proto definitions
├── src
│   ├── Cargo.toml                          <- Rust PySyft Wrapper Cargo.toml
│   ├── pyproject.toml                      <- PySyft pyproject.toml
│   ├── src
│   │   ├── README.md                       <- You are here
│   │   ├── ffi
│   │   │   ├── ffi.rs                      <- PySyft to Rust FFI Bridge
│   │   │   └── mod.rs                      <- Available FFI modules
│   │   └── lib.rs                          <- Rust lib entry point
│   ├── syft
│   │   ├── __init__.py                     <- Rust .so module imported here
│   │   └── syft.cpython-3x-platform.so     <- Compiled Rust binary for importing
│   └── target                              <- Rust build output
└── tests
    └── syft                                <- Normal Pytest tests
        ├── api
        ├── ast
        ├── core
        ├── grid
        ├── lib
        ├── notebooks
        └── rust                            <- Pytest tests that call the Rust API
```

## Setup

- python 3.6+ - https://www.python.org/
- rustup - https://rustup.rs/
- protoc - https://github.com/protocolbuffers/protobuf
- vscode - https://github.com/microsoft/vscode

### Linux

### MacOS

Python

```
$ brew install python
```

rustup

```
$ brew install rustup
$ rustup-init
```

protoc

```
$ brew install protobuf
```

### Windows

## Rust Toolchain

We are currently using stable rust.

```
$ rustup toolchain install stable
$ rustup default stable
```

### Formatting and Linting

Rust comes with an opinionated formatter and linter so we will mandate that these
are used.

Install Rust Format:

```
$ rustup component add rustfmt
```

Install Rust Language Server:

```
$ rustup component add rls
```

Install Rust Linting:

```
$ rustup component add clippy
```

### VSCode Configuration

While VSCode is not required it is highly recommended.

Install Rust VSCode Extension:
https://marketplace.visualstudio.com/items?itemName=rust-lang.rust

```
$ code --install-extension rust-lang.rust
```

Install Even Better TOML Extension:
https://marketplace.visualstudio.com/items?itemName=tamasfe.even-better-toml

```
$ code --install-extension tamasfe.even-better-toml
```

Add to settings:

```
{
  "evenBetterToml.formatter.reorderKeys": false,
  "evenBetterToml.formatter.alignEntries": true
}
```

## Python

### Setup

Make sure you have `python3.6+`

We use a virtual environment to isolate the syft core python wheel development and
build process.

We include support for Pipenv, Conda and pip with virtualenv.

### Formatting and Linting

To keep code clean and bug free we mandate all code inside syft core python, uses an
agreed upon set of linting and formatting standards.

- black - https://github.com/psf/black
- isort - https://github.com/timothycrosley/isort
- mypy - http://mypy-lang.org/

This is also checked via a pre-commit hook which is installed automatically.

### VSCode Configuration

Add these to your settings.json, making sure to update the paths as necessary to your
platform.

```
{
  "python.linting.enabled": true,
  "python.linting.mypyEnabled": true,
  "python.formatting.provider": "black",
}
```

### Python Package Managers

#### Pipenv

Upgrade pip:

```
$ pip install --upgrade pip
```

Install pipenv:

```
$ pip install pipenv
```

Create virtualenv:

```
$ pipenv --python3
$ pipenv shell
```

Install project in editable mode:

```
$ pip install -e .
```

#### Conda

Create your conda environment:

```
$ conda create --name syft --file requirements.txt
```

#### pip and virtualenv

Create a virtualenv and install the project in editable mode:

```
$ pip install -e .
```

## Maturin

We are using [maturin](https://github.com/PyO3/maturin) which is an awesome mixed python
and rust build tool.

First install it with pip:

```
$ pip install maturin
```

## Python Development

Maturin has a similar work flow to python editable mode which builds the rust code and
links it into your site-packages. You have to run it from where the `Cargo.toml` file
is located inside `PySyft/src`:

```
$ cd src
$ maturin develop
```

You should now be able to `import syft` and make live changes to the python files.
If you change Rust files you will need to run `maturin develop` again.

## Python Tests

We are using pytest which is listed in the requirements.txt.

Run tests from the root `PySyft` directory inside your virtualenv:

```
$ cd src && maturin develop; cd -
$ pytest -m fast -n auto
$ pytest -m rust -n auto
```

## Mixed Python & Rust Module Imports

The rust crate pyo3 allows us to mix compiled Rust code as a CPython module and vanilla
python code in the same wheel. The vanilla python code must go into a folder named the
same as the module and must contain at least a single `__init__.py` file in that folder.
That is why you will see this inside /platforms/python:

```
├── src             <--- Rust Code
│   └── ffi
├── syft            <--- Python Code
│   ├── ast
│   ├── core
│   ├── federated
│   ├── grid
│   ├── lib
│   ├── proto
│   └── __init__.py  <--- Where the rust module is imported
├── target
│   ├── debug
│   ├── rls
│   └── wheels
└── tests            <--- Python Tests
```

## Build Python Wheel

During this step:

- The python platform ffi code in src/src is compiled for your system arch
- The vanilla python code inside src/syft is added
- A wheel is created with both these mixed source files

Build wheel and install wheel:

```
$ maturin build -i python
$ pip install `find -L ./target/wheels -name "*.whl"`
```
