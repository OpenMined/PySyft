from __future__ import absolute_import
from distutils.version import LooseVersion
import importlib
import logging

logger = logging.getLogger(__name__)

try:
    import tensorflow

    if LooseVersion(tensorflow.__version__) < LooseVersion("2.0.0"):
        raise ImportError()
    pstf_spec = importlib.util.find_spec("syft_tensorflow")
    tensorflow_available = pstf_spec is not None
except ImportError:
    tensorflow_available = False


try:
    tfe_spec = importlib.util.find_spec("tf_encrypted")
    tfe_available = tfe_spec is not None

except ImportError as e:
    tfe_available = False


try:
    torch_spec = importlib.util.find_spec("torch")

    torch_available = torch_spec is not None

except ImportError:
    torch_available = False
