# syft-rds

The Remote Data Science client for [PySyft](https://github.com/OpenMined/PySyft) 0.10+.
It composes datasets and jobs on top of the `syft` sync engine:

```bash
pip install "syft>=0.10.0" "syft-rds>=0.6.0"
```

```python
from syft_rds import login_do, login_ds

do = login_do(email="do@org.com")     # data owner
ds = login_ds(email="ds@org.com")     # data scientist
```

See the [client API reference](https://github.com/OpenMined/PySyft/blob/dev/docs/API.md).

> **Note for SyftBox users:** versions `0.1`–`0.5` of the `syft-rds` distribution were
> the SyftBox RDS client, a different codebase. From `0.6.0` this name is the PySyft
> Remote Data Science client. If you depend on the SyftBox client, pin `syft-rds<0.6`.
