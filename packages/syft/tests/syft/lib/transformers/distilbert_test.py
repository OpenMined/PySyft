# stdlib
from typing import Any

# third party
import pytest
import torch

# syft absolute
import syft as sy
from syft import SyModule
from syft.core.plan.plan_builder import make_plan

transformers = pytest.importorskip("transformers")
distilbert = pytest.importorskip("syft.lib.transformers.models.distilbert")
sy.load("transformers")


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
def test_embedding(config: Any) -> None:
    alice = sy.VirtualMachine()
    alice_client = alice.get_root_client()

    model = distilbert.Embeddings(
        config, inputs={"input_ids": torch.ones(3, 4, dtype=torch.long)}
    )
    model.eval()
    model_ptr = model.send(alice_client)

    x = torch.LongTensor(10, 20).random_(0, 10)
    x_ptr = x.send(alice_client)

    out = model(input_ids=x)[0]
    out_remote = model_ptr(input_ids=x_ptr)[0].get()

    assert torch.equal(out, out_remote)


@pytest.mark.slow
def test_mhsa(config: Any) -> None:
    alice = sy.VirtualMachine()
    alice_client = alice.get_root_client()

    inputs = {
        "query": torch.randn(5, 10, config.dim),
        "key": torch.randn(5, 10, config.dim),
        "value": torch.randn(5, 10, config.dim),
        "mask": torch.ones(5, 10),
    }

    model = distilbert.MultiHeadSelfAttention(config, inputs=inputs)

    @make_plan
    def mhsa_forward(
        model: SyModule = model,
        query: torch.Tensor = inputs["query"],
        key: torch.Tensor = inputs["key"],
        value: torch.Tensor = inputs["value"],
        mask: torch.Tensor = inputs["mask"],
    ) -> Any:
        out = model(query=query, key=key, value=value, mask=mask)[0]
        return [out]

    inputs = {
        "query": torch.randn(8, 16, config.dim),
        "key": torch.randn(8, 16, config.dim),
        "value": torch.randn(8, 16, config.dim),
        "mask": torch.ones(8, 16),
    }
    out = model(**inputs)[0]

    fwd_ptr = mhsa_forward.send(alice_client)
    out_ptr = fwd_ptr(model=model, **inputs)

    assert torch.equal(out, out_ptr.get()[0])


@pytest.mark.slow
def test_ffn(config: Any) -> None:
    alice = sy.VirtualMachine()
    alice_client = alice.get_root_client()

    inputs = {
        "input": torch.randn(5, 10, config.dim),
    }

    model = distilbert.FFN(config, inputs=inputs)

    @make_plan
    def ffn_forward(
        model: SyModule = model, input: torch.Tensor = inputs["input"]
    ) -> Any:
        out = model(input=input)[0]
        return [out]

    inputs = {
        "input": torch.randn(8, 16, config.dim),
    }
    out = model(**inputs)[0]

    fwd_ptr = ffn_forward.send(alice_client)
    out_ptr = fwd_ptr(model=model, **inputs)[0]

    assert torch.equal(out, out_ptr.get())


@pytest.mark.slow
def test_distilbert(config: Any) -> None:
    alice = sy.VirtualMachine()
    alice_client = alice.get_root_client()

    inputs = {
        "input_ids": torch.LongTensor(5, 10).random_(0, config.vocab_size),
        "attention_mask": torch.ones(5, 10),
    }

    model = distilbert.SyDistilBert(config, inputs=inputs)

    @make_plan
    def db_forward(
        model: SyModule = model,
        input_ids: torch.Tensor = inputs["input_ids"],
        attention_mask: torch.Tensor = inputs["attention_mask"],
    ) -> Any:
        out = model(input_ids=input_ids, attention_mask=attention_mask)[0]
        return [out]

    inputs = {
        "input_ids": torch.LongTensor(8, 5).random_(0, config.vocab_size),
        "attention_mask": torch.ones(8, 5),
    }
    out = model(**inputs)[0]

    fwd_ptr = db_forward.send(alice_client)
    out_ptr = fwd_ptr(model=model, **inputs)[0]

    assert torch.equal(out, out_ptr.get())
