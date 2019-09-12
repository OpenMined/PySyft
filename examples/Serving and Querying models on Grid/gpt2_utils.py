#!/usr/bin/env python3
# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch as th
import torch.nn.functional as F


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float("Inf")):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering

        This code is from: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
        Author: Thomas Wolf.

        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """
    assert (
        logits.dim() == 1
    )  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < th.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = th.sort(logits, descending=True)
        cumulative_probs = th.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def sample_sequence(
    length,
    context,
    model=None,
    worker=None,
    model_id="GPT-2",
    num_samples=1,
    temperature=1,
    top_k=0,
    top_p=0.0,
    device="cpu",
):
    context = th.tensor(context, dtype=th.long, device=device)
    context = context.unsqueeze(0).repeat(num_samples, 1)

    predicted_indexes = []

    generated = context
    with th.no_grad():
        for _ in range(length):
            # Inference
            if worker and model_id:
                outputs = th.tensor(
                    worker.run_remote_inference(model_id=model_id, data=generated)
                )
            elif model:
                outputs = model(generated)
            else:
                raise ValueError(
                    "You should provide a worker and a model_id or a model in order to run this function."
                )

            # Applying Filter
            next_token_logits = outputs[0, -1, :] / temperature
            filtered_logits = top_k_top_p_filtering(
                next_token_logits, top_k=top_k, top_p=top_p
            )
            next_token = th.multinomial(
                F.softmax(filtered_logits, dim=-1), num_samples=1
            )

            # Update context shifting tokens
            generated = th.cat(
                (th.tensor([generated[0][1:].tolist()]), next_token.unsqueeze(0)), dim=1
            )

            # Save predicted word
            predicted_indexes.append(th.argmax(outputs[0, -1, :]).item())

    return predicted_indexes
