
Welcome to the PySyft documentation! **PySyft** is an open-source library for privacy-preserving machine learning, enabling secure data science workflows. This guide helps you set up the docs locally for development or contribution. 

### Prerequisites
- Python 3.8 or higher
- These instructions assume a Unix-like environment (Linux/Mac or WSL on Windows). On    Windows with PowerShell, you may need to install tools like `make` via Chocolatey or adapt commands accordingly.



# PySyft Documentation

Welcome to the PySyft docs. You can setup the PySyft docs locally via 2 methods currently,

- Natively using `sphinx-apidoc` command
- Using `tox` command (this is what we also use for our deployments)

## Setting it up natively

1. Install dependencies:

   ```sh
   cd docs
   pip install -r requirements.txt
   ```

2. Get into the source subdirectory now and generate `sphinx-apidoc`:

   ```sh
   cd source
   sphinx-apidoc -f -M -d 2 -o ./api_reference/ ../../packages/syft/src/syft
   ```

3. Now go back one directory up and generate HTML docs:

   ```sh
   cd ../
   make html
   ```

4. Voila! Now visit the PySyft/docs/build/html/index.html to view the docs locally

## Setting it up using Tox

1. Install tox:

   ```sh
    pip install tox
   ```

2. Run the following command:

   ```sh
   tox -e syft.docs
   ```

3. Voila! Now visit the PySyft/docs/build/html/index.html to view the docs locally.

## Viewing the Docs

After running either setup method, open `docs/build/html/index.html` in your web browser to view the generated documentation locally.

## Debugging

If you want to start a fresh build, run:

```sh
make clean
