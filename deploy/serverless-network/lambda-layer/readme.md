### Create Lambda Layer package

Let us first go to `apps/network` and export the poetry lock file to a requirements file.
```shell script
cd ../../../apps/network
poetry export --format requirements.txt -o ../../deploy/serverless-network/lambda-layer/requirements.txt --without-hashes
cd ../../deploy/serverless-network/lambda-layer/
```
Build a zip file containing all dependencies of PyGrid Network, to deploy to an AWS Lambda Layer.
The root file should be called `Python`, and contains all the dependencies.

```shell script
mkdir python
pip install -r requirements.txt -t python
zip -r all-dep.zip python
```
Remove the temporary files and folders.
```shell script
rm -rf python requirements.txt
```