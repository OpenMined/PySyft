# third party
import pytest
import torch

# syft absolute
import syft as sy
from syft.lib.python import List

transformers = pytest.importorskip("transformers")


@pytest.mark.slow
def test_tokenizer_serde() -> None:
    alice = sy.VirtualMachine()
    alice_client = alice.get_root_client()

    tokenizer = transformers.AutoTokenizer.from_pretrained("distilbert-base-uncased")

    tok_ptr = tokenizer.send(alice_client)
    tok_serde = tok_ptr.get()

    assert tok_serde._tokenizer.to_str() == tokenizer._tokenizer.to_str()
    assert tok_serde.name_or_path == tokenizer.name_or_path
    assert tok_serde.padding_side == tokenizer.padding_side
    assert tok_serde.model_max_length == tokenizer.model_max_length
    assert tok_serde.special_tokens_map == tokenizer.special_tokens_map


@pytest.mark.slow
def test_batchencoding_serde() -> None:
    alice = sy.VirtualMachine()
    alice_client = alice.get_root_client()

    tokenizer = transformers.AutoTokenizer.from_pretrained("distilbert-base-uncased")

    tok_ptr = tokenizer.send(alice_client)

    batch = ["This is a test", "Another test!"]
    enc = tokenizer(batch, padding=True, return_tensors="pt", truncation=True)

    batch_ptr = List(batch).send(alice_client)
    enc_ptr = tok_ptr(batch_ptr, padding=True, return_tensors="pt", truncation=True)

    enc_remote = enc_ptr.get()

    assert torch.equal(enc_remote["input_ids"], enc["input_ids"])
    assert torch.equal(enc_remote["attention_mask"], enc["attention_mask"])
