import logging

logger = logging.getLogger(__name__)

try:
    import tf_encrypted as tfe

    keras_available = True
except ImportError as e:
    keras_available = False
