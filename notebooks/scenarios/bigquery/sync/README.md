# Testing works over 4 possibilities

1. (python/in-memory workers and using tox commands)
2. (python/in-memory workers and manually running notebooks)
3. (using k8s and using tox commands)
4. (using k8s and manually running notebooks)

Add the lines below to notebook cells if in the 4th possibility

```python
os.environ["ORCHESTRA_DEPLOYMENT_TYPE"] = "remote"
os.environ["DEV_MODE"] = "True"
os.environ["TEST_EXTERNAL_REGISTRY"] = "k3d-registry.localhost:5800"
os.environ["CLUSTER_HTTP_PORT_HIGH"] = "9081"
os.environ["CLUSTER_HTTP_PORT_LOW"] = "9083"
```
