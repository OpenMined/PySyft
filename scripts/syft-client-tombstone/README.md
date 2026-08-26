# `syft-client` has been renamed to `syft`

This project is retired. Development continues as [`syft`](https://pypi.org/project/syft/)
in the [OpenMined/PySyft](https://github.com/OpenMined/PySyft) repository, starting at
`syft==0.10.0`.

```bash
pip install -U syft               # data scientists
pip install -U syft syft-bg syft-job   # data owners running syft-bg (stop the services first)
```

```python
import syft as sy   # was: import syft_client as sc
```

`syft-client==0.2.0` exists only to redirect: it depends on `syft>=0.10.0` and
`import syft_client` raises an `ImportError` pointing here.

---

Maintainers: this directory is the source of that final release. It is deliberately
outside `packages/` (not a uv workspace member) and outside the root package
discovery. Publish once, after `syft==0.10.0` is live on PyPI:

```bash
cd scripts/syft-client-tombstone
rm -rf dist && uv build && uvx twine upload dist/*
```
