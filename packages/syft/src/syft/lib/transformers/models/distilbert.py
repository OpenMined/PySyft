# stdlib
import math
import time

# third party
import torch
from torch import nn
from transformers import AutoModel
from transformers.models.distilbert.modeling_distilbert import (
    create_sinusoidal_embeddings,
)
from transformers.models.distilbert.modeling_distilbert import DistilBertConfig

# syft absolute
from syft import SyModule
from syft.core.plan.plan_builder import ROOT_CLIENT


class Embeddings(SyModule):
    def __init__(self, config: DistilBertConfig, **kwargs) -> None:  # type: ignore
        super().__init__(**kwargs)
        self.config = config
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.dim, padding_idx=config.pad_token_id
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.dim
        )

        if config.sinusoidal_pos_embds:
            create_sinusoidal_embeddings(
                n_pos=config.max_position_embeddings,
                dim=config.dim,
                out=self.position_embeddings.weight,
            )

        self.LayerNorm = nn.LayerNorm(config.dim, eps=1e-12)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids: torch.LongTensor) -> torch.FloatTensor:
        remote_torch = ROOT_CLIENT.torch

        seq_length = input_ids.size(1)
        # TODO setting device from input_ids from remotely created tensor throws KeyError: UID <...>.
        position_ids = remote_torch.arange(seq_length)  # (max_seq_length)
        position_ids = remote_torch.unsqueeze(position_ids, 0).expand(
            input_ids.size(0), input_ids.size(1)
        )  # (bs, max_seq_length)

        word_embeddings = self.word_embeddings(input_ids)  # (bs, max_seq_length, dim)
        position_embeddings = self.position_embeddings(
            position_ids
        )  # (bs, max_seq_length, dim)

        embeddings = word_embeddings + position_embeddings  # (bs, max_seq_length, dim)
        embeddings = self.LayerNorm(embeddings)  # (bs, max_seq_length, dim)
        embeddings = self.dropout(embeddings)  # (bs, max_seq_length, dim)
        return embeddings


class MultiHeadSelfAttention(SyModule):
    def __init__(self, config: DistilBertConfig, **kwargs) -> None:  # type: ignore
        super().__init__(**kwargs)

        if config.dim % config.n_heads != 0:
            raise ValueError("`config.dim` should be a multiple of `config.n_heads`")

        self.n_heads = config.n_heads
        self.dim = config.dim
        self.dropout = nn.Dropout(p=config.attention_dropout)

        self.q_lin = nn.Linear(in_features=config.dim, out_features=config.dim)
        self.k_lin = nn.Linear(in_features=config.dim, out_features=config.dim)
        self.v_lin = nn.Linear(in_features=config.dim, out_features=config.dim)
        self.out_lin = nn.Linear(in_features=config.dim, out_features=config.dim)

    def shape(self, x: torch.Tensor, bs: int, dim_per_head: int) -> torch.Tensor:
        """separate heads for linear layers"""
        return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)

    def unshape(self, x: torch.Tensor, bs: int, dim_per_head: int) -> torch.Tensor:
        """group heads"""
        return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters:
            query: torch.tensor(bs, seq_length, dim)
            key: torch.tensor(bs, seq_length, dim)
            value: torch.tensor(bs, seq_length, dim)
            mask: torch.tensor(bs, seq_length)
        Returns:
            weights: torch.tensor(bs, n_heads, seq_length, seq_length)
        """
        remote_torch = ROOT_CLIENT.torch

        bs = query.size(0)
        k_length = key.size(1)

        dim_per_head = self.dim // self.n_heads
        mask_reshp = (bs, 1, 1, k_length)

        # Linear
        q = self.shape(
            self.q_lin(query), bs, dim_per_head
        )  # (bs, n_heads, q_length, dim_per_head)
        k = self.shape(
            self.k_lin(key), bs, dim_per_head
        )  # (bs, n_heads, k_length, dim_per_head)
        v = self.shape(
            self.v_lin(value), bs, dim_per_head
        )  # (bs, n_heads, k_length, dim_per_head)

        q = q / math.sqrt(dim_per_head)  # (bs, n_heads, q_length, dim_per_head)
        scores = remote_torch.matmul(
            q, k.transpose(2, 3)
        )  # (bs, n_heads, q_length, k_length)

        mask = (mask == 0).view(*mask_reshp)
        mask = mask.expand(-1, scores.size(1), scores.size(2), -1)

        # TODO this is equivalent to the two above lines, expand_as does not work.
        # mask = (mask == 0).view(*mask_reshp).expand_as(scores)

        # Softmax
        scores.masked_fill_(mask, -float("inf"))  # (bs, n_heads, q_length, k_length)
        weights = remote_torch.softmax(
            scores, dim=-1
        )  # (bs, n_heads, q_length, k_length)
        weights = self.dropout(weights)  # (bs, n_heads, q_length, k_length)

        context = remote_torch.matmul(
            weights, v
        )  # (bs, n_heads, q_length, dim_per_head)
        context = self.unshape(context, bs, dim_per_head)  # (bs, q_length, dim)
        context = self.out_lin(context)  # (bs, q_length, dim)

        return context


class FFN(SyModule):
    def __init__(self, config: DistilBertConfig, **kwargs) -> None:  # type: ignore
        super().__init__(**kwargs)
        self.dropout = nn.Dropout(p=config.dropout)
        self.seq_len_dim = 1
        self.lin1 = nn.Linear(in_features=config.dim, out_features=config.hidden_dim)
        self.lin2 = nn.Linear(in_features=config.hidden_dim, out_features=config.dim)

        if config.activation == "gelu":
            self.activation = ROOT_CLIENT.torch.nn.functional.gelu
        elif config.activation == "relu":
            self.activation = ROOT_CLIENT.torch.nn.functional.relu
        else:
            raise ValueError(
                f"activation ({config.activation}) must be in ['relu', 'gelu']"
            )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.lin1(input)
        x = self.activation(x)
        x = self.lin2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(SyModule):
    def __init__(self, config: DistilBertConfig, **kwargs) -> None:  # type: ignore
        super().__init__(**kwargs)
        if config.dim % config.n_heads != 0:
            raise ValueError("`config.dim` should be a multiple of `config.n_heads`")

        attn_dummy_inputs = {
            "query": kwargs["inputs"]["x"],
            "key": kwargs["inputs"]["x"],
            "value": kwargs["inputs"]["x"],
            "mask": kwargs["inputs"]["attn_mask"],
        }
        self.attention = MultiHeadSelfAttention(config, inputs=attn_dummy_inputs)

        ffn_dummy_inputs = {"input": kwargs["inputs"]["x"]}
        self.ffn = FFN(config, inputs=ffn_dummy_inputs)

        self.sa_layer_norm = nn.LayerNorm(normalized_shape=config.dim, eps=1e-12)
        self.output_layer_norm = nn.LayerNorm(normalized_shape=config.dim, eps=1e-12)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        # Self-Attention
        sa_output = self.attention(query=x, key=x, value=x, mask=attn_mask)[0]

        sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)

        # Feed Forward Network
        ffn_output = self.ffn(input=sa_output)[0]  # (bs, seq_length, dim)
        ffn_output = self.output_layer_norm(
            ffn_output + sa_output
        )  # (bs, seq_length, dim)

        return ffn_output


class Transformer(SyModule):
    def __init__(self, config: DistilBertConfig, **kwargs) -> None:  # type: ignore

        t0 = time.time()

        super().__init__(**kwargs)
        self.n_layers = config.n_layers

        self.layer = nn.ModuleList(
            [
                TransformerBlock(config, inputs=kwargs["inputs"])
                for _ in range(self.n_layers)
            ]
        )

        print(f"transformer init: {time.time() - t0:.2f} s")

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """
        - SyModule does not work; multiple inputs
        - getattr does not work;
        """
        t0 = time.time()
        hidden_state = x

        # TODO fix ModuleList.__iter__; items in iter need to be ModulePointer
        for i in range(self.n_layers):
            layer = self.layer[i]
            hidden_state = layer(x=hidden_state, attn_mask=attn_mask)[0]

        print(f"transformer forward: {time.time() - t0:.2f} s")
        return hidden_state


class SyDistilBert(SyModule):
    def __init__(self, config: DistilBertConfig, **kwargs) -> None:  # type: ignore
        """
        SyDistilBert is a re-implementation of huggingface DistilBert in pysyft,
        with all non-torch-native submodules rewritten as SyModules.

        Use the `from_pretrained` or `from_config` classmethods to instantiate this
        model from an existing HuggingFace pretrained model or config.
        """
        super().__init__(**kwargs)
        self.config = config

        # Embeddings
        embedding_inputs = {"input_ids": kwargs["inputs"]["input_ids"]}
        self.embeddings = Embeddings(config=config, inputs=embedding_inputs)

        # Transformer
        transformer_x = torch.rand(*kwargs["inputs"]["input_ids"].size(), config.dim)
        transformer_mask = kwargs["inputs"]["attention_mask"]
        transformer_inputs = {"x": transformer_x, "attn_mask": transformer_mask}

        self.transformer = Transformer(config=config, inputs=transformer_inputs)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        # [B, S] -> [B, S, D]
        input_embeds = self.embeddings(input_ids=input_ids)[0]

        # [B, S, D] -> [B, S, D]
        out = self.transformer(x=input_embeds, attn_mask=attention_mask)[0]
        return out

    @classmethod
    def from_pretrained(cls, model_name: str) -> "SyDistilBert":
        # Make dummy inputs
        dummy_x = torch.ones(2, 3, dtype=torch.long)
        dummy_mask = torch.ones(2, 3, dtype=torch.long)

        dummy_inputs = {"input_ids": dummy_x, "attention_mask": dummy_mask}

        # Load huggingface model
        hf_model = AutoModel.from_pretrained(model_name)

        # Construct model
        model = cls(hf_model.config, inputs=dummy_inputs)

        # Load weights
        model.load_state_dict(hf_model.state_dict())

        return model

    @classmethod
    def from_config(cls, config: DistilBertConfig) -> "SyDistilBert":
        # Make dummy inputs
        dummy_x = torch.ones(2, 3, dtype=torch.long)
        dummy_mask = torch.ones(2, 3, dtype=torch.long)

        dummy_inputs = {"input_ids": dummy_x, "attention_mask": dummy_mask}

        # Construct model
        model = cls(config, inputs=dummy_inputs)

        return model
