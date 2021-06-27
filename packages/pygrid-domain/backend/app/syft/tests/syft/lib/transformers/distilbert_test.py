# stdlib
from typing import Any

# third party
import pytest
import torch

# syft absolute
import syft as sy
from syft.core.plan.plan_builder import make_plan

transformers = pytest.importorskip("transformers")
distilbert = pytest.importorskip("syft.lib.transformers.models.distilbert")


@pytest.fixture(scope="module")
def config() -> Any:
    return transformers.DistilBertConfig(
        vocab_size=10,
        dim=16,
        max_position_embeddings=100,
        n_heads=2,
        hidden_dim=10,
        n_layers=2,
        dropout=0,
        attention_dropout=0,
    )


@pytest.mark.slow
def test_distilbert(config: Any) -> None:
    alice = sy.VirtualMachine()
    alice_client = alice.get_root_client()

    inputs = {
        "input_ids": torch.LongTensor(5, 10).random_(0, config.vocab_size),
        "attention_mask": torch.ones(5, 10),
    }

    model = transformers.DistilBertModel(config)

    @make_plan
    def db_forward(
        model: torch.nn.Module = model,
        input_ids: torch.Tensor = inputs["input_ids"],
        attention_mask: torch.Tensor = inputs["attention_mask"],
    ) -> Any:
        out = model(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=False
        )
        return [out]

    inputs = {
        "input_ids": torch.LongTensor(8, 5).random_(0, config.vocab_size),
        "attention_mask": torch.ones(8, 5),
    }
    out = model(**inputs)[0]

    fwd_ptr = db_forward.send(alice_client)
    out_ptr = fwd_ptr(model=model, **inputs)[0]

    assert torch.equal(out, out_ptr.get()[0])
