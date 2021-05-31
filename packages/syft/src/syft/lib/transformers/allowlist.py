# stdlib
from typing import Dict
from typing import Union

# syft relative
from ..misc.union import UnionGenerator

allowlist: Dict[str, Union[str, Dict[str, str]]] = {} # (path: str, return_type:type)

# DistilBert top level modules

allowlist["transformers.models.distilbert.modeling_distilbert.Embeddings"] = "transformers.models.distilbert.modeling_distilbert.Embeddings"
allowlist["transformers.models.distilbert.modeling_distilbert.Embeddings.__call__"] = "torch.Tensor"
allowlist["transformers.models.distilbert.modeling_distilbert.Embeddings.parameters"] = "syft.lib.python.List"
allowlist["transformers.models.distilbert.modeling_distilbert.Embeddings.register_parameter"] = "syft.lib.python._SyNone"
allowlist["transformers.models.distilbert.modeling_distilbert.Embeddings.register_parameter"] = "syft.lib.python._SyNone"
allowlist["transformers.models.distilbert.modeling_distilbert.Embeddings.train"] = "transformers.models.distilbert.modeling_distilbert.Embeddings"
allowlist["transformers.models.distilbert.modeling_distilbert.Embeddings.cuda"] = "transformers.models.distilbert.modeling_distilbert.Embeddings"
allowlist["transformers.models.distilbert.modeling_distilbert.Embeddings.cpu"] = "transformers.models.distilbert.modeling_distilbert.Embeddings"
allowlist["transformers.models.distilbert.modeling_distilbert.Embeddings.state_dict"] = "syft.lib.python.collections.OrderedDict"
allowlist["transformers.models.distilbert.modeling_distilbert.Embeddings.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["transformers.models.distilbert.modeling_distilbert.Embeddings.extra_repr"] = "syft.lib.python.String"

allowlist["transformers.models.distilbert.modeling_distilbert.Transformer"] = "transformers.models.distilbert.modeling_distilbert.Transformer"
allowlist["transformers.models.distilbert.modeling_distilbert.Transformer.__call__"] = "syft.lib.python.Tuple"
allowlist["transformers.models.distilbert.modeling_distilbert.Transformer.parameters"] = "syft.lib.python.List"
allowlist["transformers.models.distilbert.modeling_distilbert.Transformer.register_parameter"] = "syft.lib.python._SyNone"
allowlist["transformers.models.distilbert.modeling_distilbert.Transformer.register_parameter"] = "syft.lib.python._SyNone"
allowlist["transformers.models.distilbert.modeling_distilbert.Transformer.train"] = "transformers.models.distilbert.modeling_distilbert.Transformer"
allowlist["transformers.models.distilbert.modeling_distilbert.Transformer.cuda"] = "transformers.models.distilbert.modeling_distilbert.Transformer"
allowlist["transformers.models.distilbert.modeling_distilbert.Transformer.cpu"] = "transformers.models.distilbert.modeling_distilbert.Transformer"
allowlist["transformers.models.distilbert.modeling_distilbert.Transformer.state_dict"] = "syft.lib.python.collections.OrderedDict"
allowlist["transformers.models.distilbert.modeling_distilbert.Transformer.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["transformers.models.distilbert.modeling_distilbert.Transformer.extra_repr"] = "syft.lib.python.String"

allowlist["transformers.models.distilbert.modeling_distilbert.DistilBertModel"] = "transformers.models.distilbert.modeling_distilbert.DistilBertModel"
allowlist["transformers.models.distilbert.modeling_distilbert.DistilBertModel.__call__"] = "syft.lib.python.Tuple"
allowlist["transformers.models.distilbert.modeling_distilbert.DistilBertModel.parameters"] = "syft.lib.python.List"
allowlist["transformers.models.distilbert.modeling_distilbert.DistilBertModel.register_parameter"] = "syft.lib.python._SyNone"
allowlist["transformers.models.distilbert.modeling_distilbert.DistilBertModel.register_parameter"] = "syft.lib.python._SyNone"
allowlist["transformers.models.distilbert.modeling_distilbert.DistilBertModel.train"] = "transformers.models.distilbert.modeling_distilbert.Transformer"
allowlist["transformers.models.distilbert.modeling_distilbert.DistilBertModel.cuda"] = "transformers.models.distilbert.modeling_distilbert.Transformer"
allowlist["transformers.models.distilbert.modeling_distilbert.DistilBertModel.cpu"] = "transformers.models.distilbert.modeling_distilbert.Transformer"
allowlist["transformers.models.distilbert.modeling_distilbert.DistilBertModel.state_dict"] = "syft.lib.python.collections.OrderedDict"
allowlist["transformers.models.distilbert.modeling_distilbert.DistilBertModel.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["transformers.models.distilbert.modeling_distilbert.DistilBertModel.extra_repr"] = "syft.lib.python.String"