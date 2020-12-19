# stdlib
from typing import Dict
from typing import Union

# This is the SyferText allowlist.
# In order to use any module, class, function or method in SyferText,
# a dedicated entry should be created in this dictionary.
# The key of each entry is the fully qualified name (aka. path) of that
# module, class, function or method. The value could be either
# a string specifying the fully qualified name (aka. path) of the
# return type, or a dictionary of the form:
#
# {
#   'return_type' : "<path>",
#   'min_versoin' : "<minimum SyferText version to support this element>",
#   'max_version' : "<maximum SyferText version to support this element>"
# }
#
# 'min_version' and 'max_version' in the above dictionary are optional
#
allowlist: Dict[str, Union[str, Dict[str, str]]] = {} 

#allowlist["syfertext"] = "syfertext"

# --------------------------------------------------------------------------------------
# SECTION - Tokenizers
# --------------------------------------------------------------------------------------

allowlist["syfertext.tokenizers"] = "syfertext.tokenizers"
allowlist["syfertext.tokenizers.default_tokenizer"] = "syfertext.tokenizers.default_tokenizer"
allowlist["syfertext.tokenizers.default_tokenizer.DefaultTokenizer"] = "syfertext.tokenizers.default_tokenizer.DefaultTokenizer"
