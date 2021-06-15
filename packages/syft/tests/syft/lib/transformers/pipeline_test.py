# third party
import pytest

# syft absolute
import syft as sy

# @pytest.mark.vendor(lib="transformers")
# def test_pipeline(
#     root_client: sy.VirtualMachineClient,
# ) -> None:
#     # third party
#     import transformers

#     pipeline = root_client.transformers.pipelines.pipeline
#     generator = pipeline("text-generation", model="gpt2")
#     results = generator(
#         "Hello, I'm a language model,", max_length=30, num_return_sequences=5
#     )

#     assert len(results.get()) == 5


# @pytest.mark.vendor(lib="transformers")
# def test_gpt2() -> None:
#     pass
#     # TODO: serde these
#     # transformers.models.gpt2.configuration_gpt2.GPT2Config
#     # transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel
#     # transformers.models.gpt2.tokenization_gpt2_fast.GPT2TokenizerFast
