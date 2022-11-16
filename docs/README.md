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

## Debugging

If you want to start a fresh build, run:

```sh
make clean
```

as this will remove all the pre-exisitng files or directories in the build/ directory.
