# stdlib
from typing import Dict
from typing import Union

allowlist: Dict[str, Union[str, Dict[str, str]]] = {}  # (path: str, return_type:type)


# General
allowlist[
    "transformers.tokenization_utils_base.BatchEncoding"
] = "transformers.tokenization_utils_base.BatchEncoding"
allowlist[
    "transformers.tokenization_utils_base.BatchEncoding.__getitem__"
] = "torch.Tensor"

# Distilbert model
allowlist[
    "transformers.models.distilbert.modeling_distilbert.DistilBertModel"
] = "transformers.models.distilbert.modeling_distilbert.DistilBertModel"
allowlist[
    "transformers.models.distilbert.modeling_distilbert.DistilBertModel.__call__"
] = "torch.Tensor"
allowlist[
    "transformers.models.distilbert.modeling_distilbert.DistilBertModel.parameters"
] = "syft.lib.python.List"
allowlist[
    "transformers.models.distilbert.modeling_distilbert.DistilBertModel.register_parameter"
] = "syft.lib.python._SyNone"
allowlist[
    "transformers.models.distilbert.modeling_distilbert.DistilBertModel.register_parameter"
] = "syft.lib.python._SyNone"
allowlist[
    "transformers.models.distilbert.modeling_distilbert.DistilBertModel.train"
] = "transformers.models.distilbert.modeling_distilbert.DistilBertModel"
allowlist[
    "transformers.models.distilbert.modeling_distilbert.DistilBertModel.eval"
] = "transformers.models.distilbert.modeling_distilbert.DistilBertModel"
allowlist[
    "transformers.models.distilbert.modeling_distilbert.DistilBertModel.cuda"
] = "transformers.models.distilbert.modeling_distilbert.DistilBertModel"
allowlist[
    "transformers.models.distilbert.modeling_distilbert.DistilBertModel.cpu"
] = "transformers.models.distilbert.modeling_distilbert.DistilBertModel"
allowlist[
    "transformers.models.distilbert.modeling_distilbert.DistilBertModel.state_dict"
] = "syft.lib.python.collections.OrderedDict"
allowlist[
    "transformers.models.distilbert.modeling_distilbert.DistilBertModel.load_state_dict"
] = "syft.lib.python._SyNone"
allowlist[
    "transformers.models.distilbert.modeling_distilbert.DistilBertModel.extra_repr"
] = "syft.lib.python.String"

# DistilBert modules
allowlist[
    "transformers.models.distilbert.modeling_distilbert.Embeddings"
] = "transformers.models.distilbert.modeling_distilbert.Embeddings"
allowlist[
    "transformers.models.distilbert.modeling_distilbert.MultiHeadSelfAttention"
] = "transformers.models.distilbert.modeling_distilbert.MultiHeadSelfAttention"
allowlist[
    "transformers.models.distilbert.modeling_distilbert.FFN"
] = "transformers.models.distilbert.modeling_distilbert.FFN"
allowlist[
    "transformers.models.distilbert.modeling_distilbert.TransformerBlock"
] = "transformers.models.distilbert.modeling_distilbert.TransformerBlock"
allowlist[
    "transformers.models.distilbert.modeling_distilbert.Transformer"
] = "transformers.models.distilbert.modeling_distilbert.Transformer"

# DistilBert misc
allowlist[
    "transformers.models.distilbert.configuration_distilbert.DistilBertConfig"
] = "transformers.models.distilbert.configuration_distilbert.DistilBertConfig"
allowlist[
    "transformers.models.distilbert.tokenization_distilbert_fast.DistilBertTokenizerFast"
] = "transformers.models.distilbert.tokenization_distilbert_fast.DistilBertTokenizerFast"
allowlist[
    "transformers.models.distilbert.tokenization_distilbert_fast.DistilBertTokenizerFast.__call__"
] = "transformers.tokenization_utils_base.BatchEncoding"
