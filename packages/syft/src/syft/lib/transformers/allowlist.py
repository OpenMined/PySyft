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
allowlist[
    "transformers.tokenization_utils_fast.PreTrainedTokenizerFast"
] = "transformers.tokenization_utils_fast.PreTrainedTokenizerFast"
allowlist[
    "transformers.tokenization_utils_fast.PreTrainedTokenizerFast.__call__"
] = "transformers.tokenization_utils_base.BatchEncoding"


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

# XLMRoberta
allowlist[
    "transformers.models.xlm_roberta.modeling_xlm_roberta.XLMRobertaModel"
] = "transformers.models.xlm_roberta.modeling_xlm_roberta.XLMRobertaModel"
allowlist[
    "transformers.models.xlm_roberta.modeling_xlm_roberta.XLMRobertaModel.from_pretrained"
] = "transformers.models.xlm_roberta.modeling_xlm_roberta.XLMRobertaModel"
allowlist[
    "transformers.models.xlm_roberta.modeling_xlm_roberta.XLMRobertaModel.__call__"
] = "torch.Tensor"
allowlist[
    "transformers.models.xlm_roberta.modeling_xlm_roberta.XLMRobertaModel.parameters"
] = "syft.lib.python.List"
allowlist[
    "transformers.models.xlm_roberta.modeling_xlm_roberta.XLMRobertaModel.register_parameter"
] = "syft.lib.python._SyNone"
allowlist[
    "transformers.models.xlm_roberta.modeling_xlm_roberta.XLMRobertaModel.register_parameter"
] = "syft.lib.python._SyNone"
allowlist[
    "transformers.models.xlm_roberta.modeling_xlm_roberta.XLMRobertaModel.train"
] = "transformers.models.xlm_roberta.modeling_xlm_roberta.XLMRobertaModel"
allowlist[
    "transformers.models.xlm_roberta.modeling_xlm_roberta.XLMRobertaModel.eval"
] = "transformers.models.xlm_roberta.modeling_xlm_roberta.XLMRobertaModel"
allowlist[
    "transformers.models.xlm_roberta.modeling_xlm_roberta.XLMRobertaModel.cuda"
] = "transformers.models.xlm_roberta.modeling_xlm_roberta.XLMRobertaModel"
allowlist[
    "transformers.models.xlm_roberta.modeling_xlm_roberta.XLMRobertaModel.cpu"
] = "transformers.models.xlm_roberta.modeling_xlm_roberta.XLMRobertaModel"
allowlist[
    "transformers.models.xlm_roberta.modeling_xlm_roberta.XLMRobertaModel.state_dict"
] = "syft.lib.python.collections.OrderedDict"
allowlist[
    "transformers.models.xlm_roberta.modeling_xlm_roberta.XLMRobertaModel.load_state_dict"
] = "syft.lib.python._SyNone"
allowlist[
    "transformers.models.xlm_roberta.modeling_xlm_roberta.XLMRobertaModel.extra_repr"
] = "syft.lib.python.String"

# Roberta classification
allowlist[
    "transformers.models.roberta.modeling_roberta.RobertaClassificationHead"
] = "transformers.models.roberta.modeling_roberta.RobertaClassificationHead"
allowlist[
    "transformers.models.roberta.modeling_roberta.RobertaClassificationHead.__call__"
] = "torch.Tensor"
allowlist[
    "transformers.models.roberta.modeling_roberta.RobertaClassificationHead.parameters"
] = "syft.lib.python.List"
allowlist[
    "transformers.models.roberta.modeling_roberta.RobertaClassificationHead.register_parameter"
] = "syft.lib.python._SyNone"
allowlist[
    "transformers.models.roberta.modeling_roberta.RobertaClassificationHead.register_parameter"
] = "syft.lib.python._SyNone"
allowlist[
    "transformers.models.roberta.modeling_roberta.RobertaClassificationHead.train"
] = "transformers.models.roberta.modeling_roberta.RobertaClassificationHead"
allowlist[
    "transformers.models.roberta.modeling_roberta.RobertaClassificationHead.eval"
] = "transformers.models.roberta.modeling_roberta.RobertaClassificationHead"
allowlist[
    "transformers.models.roberta.modeling_roberta.RobertaClassificationHead.cuda"
] = "transformers.models.roberta.modeling_roberta.RobertaClassificationHead"
allowlist[
    "transformers.models.roberta.modeling_roberta.RobertaClassificationHead.cpu"
] = "transformers.models.roberta.modeling_roberta.RobertaClassificationHead"
allowlist[
    "transformers.models.roberta.modeling_roberta.RobertaClassificationHead.state_dict"
] = "syft.lib.python.collections.OrderedDict"
allowlist[
    "transformers.models.roberta.modeling_roberta.RobertaClassificationHead.load_state_dict"
] = "syft.lib.python._SyNone"
allowlist[
    "transformers.models.roberta.modeling_roberta.RobertaClassificationHead.extra_repr"
] = "syft.lib.python.String"


# Roberta submodules
allowlist[
    "transformers.models.roberta.modeling_roberta.RobertaEmbeddings"
] = "transformers.models.roberta.modeling_roberta.RobertaEmbeddings"

allowlist[
    "transformers.models.roberta.modeling_roberta.RobertaSelfAttention"
] = "transformers.models.roberta.modeling_roberta.RobertaSelfAttention"

allowlist[
    "transformers.models.roberta.modeling_roberta.RobertaSelfOutput"
] = "transformers.models.roberta.modeling_roberta.RobertaSelfOutput"

allowlist[
    "transformers.models.roberta.modeling_roberta.RobertaAttention"
] = "transformers.models.roberta.modeling_roberta.RobertaAttention"

allowlist[
    "transformers.models.roberta.modeling_roberta.RobertaIntermediate"
] = "transformers.models.roberta.modeling_roberta.RobertaIntermediate"

allowlist[
    "transformers.models.roberta.modeling_roberta.RobertaOutput"
] = "transformers.models.roberta.modeling_roberta.RobertaOutput"

allowlist[
    "transformers.models.roberta.modeling_roberta.RobertaLayer"
] = "transformers.models.roberta.modeling_roberta.RobertaLayer"

allowlist[
    "transformers.models.roberta.modeling_roberta.RobertaEncoder"
] = "transformers.models.roberta.modeling_roberta.RobertaEncoder"

allowlist[
    "transformers.models.roberta.modeling_roberta.RobertaPooler"
] = "transformers.models.roberta.modeling_roberta.RobertaPooler"

allowlist[
    "transformers.models.roberta.modeling_roberta.RobertaClassificationHead"
] = "transformers.models.roberta.modeling_roberta.RobertaClassificationHead"

# XLMRoberta misc

allowlist[
    "transformers.models.xlm_roberta.configuration_xlm_roberta.XLMRobertaConfig"
] = "transformers.models.xlm_roberta.configuration_xlm_roberta.XLMRobertaConfig"

allowlist[
    "transformers.models.xlm_roberta.tokenization_xlm_roberta_fast.XLMRobertaTokenizerFast"
] = "transformers.models.xlm_roberta.tokenization_xlm_roberta_fast.XLMRobertaTokenizerFast"

allowlist[
    "transformers.models.xlm_roberta.tokenization_xlm_roberta_fast.XLMRobertaTokenizerFast.from_pretrained"
] = "transformers.models.xlm_roberta.tokenization_xlm_roberta_fast.XLMRobertaTokenizerFast"

allowlist[
    "transformers.models.xlm_roberta.tokenization_xlm_roberta_fast.XLMRobertaTokenizerFast.__call__"
] = "transformers.tokenization_utils_base.BatchEncoding"
