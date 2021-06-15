# stdlib
from pathlib import Path
import platform
import tarfile
import tempfile
from typing import Optional
from typing import Union

# third party
import numpy as np
import torch as th


class Dataset:
    def __init__(
        self,
        data: Union[th.Tensor, list, np.ndarray],
        description: Optional[str] = None,
        tags: Optional[list] = None,
    ) -> None:
        """WARNING: this should only ever be callable on a CLIENT
        by a DATA OWNER. This class should NEVER BE DESERIALIZED ONTO A WORKER.
        It INTERACTS WITH THE FILE SYSTEM!
        This is a convenience class just to make uploading datasets to PyGrid
        easier to do from a notebook until we have the ability to upload directly.
        """
        if isinstance(data, th.Tensor):
            data = data.numpy()
        elif isinstance(data, list):
            data = np.array(data)
        self.data = data
        self.desc = description
        self.tags = tags

    def tozip(self) -> Path:
        tempdir = Path(
            "/tmp" if platform.system() == "Darwin" else tempfile.gettempdir()
        )
        tempdir = str(tempdir.absolute())
        np.savetxt(tempdir + "/data_01.csv", self.data, delimiter=",")
        f = open(tempdir + "/manifest", "w")
        f.write(self.desc)
        f.close()
        f = open(tempdir + "/description", "w")
        f.write(self.desc)
        f.close()
        f = open(tempdir + "/tags", "w")
        for tag in self.tags:
            f.write(tag + "\n")
        f.close()
        tar = tarfile.open(tempdir + "/sample.tar.gz", "w:gz")
        for name in [
            tempdir + "/data_01.csv",
            tempdir + "/manifest",
            tempdir + "/description",
            tempdir + "/tags",
        ]:
            tar.add(name)
        tar.close()
        return tempdir + "/sample.tar.gz"
