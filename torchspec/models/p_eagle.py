# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from torchspec.models.eagle3 import Eagle3Model, LazyTarget, PrecomputedTarget


@dataclass
class PEagleInputs:
    input_ids: torch.Tensor
    hidden_states: torch.Tensor
    loss_mask: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    depth_ids: torch.Tensor


def build_p_eagle_inputs(
    input_ids: torch.Tensor,
    hidden_states: torch.Tensor,
    loss_mask: torch.Tensor,
    depth: int,
    mask_token_id: int,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
) -> PEagleInputs:
    bsz, seq_len = input_ids.shape
    device = input_ids.device

    depth_ids = torch.arange(depth, device=device).view(1, 1, depth).expand(bsz, seq_len, depth)
    expanded_input_ids = input_ids.unsqueeze(-1).expand(bsz, seq_len, depth).clone()
    if depth > 1:
        expanded_input_ids[:, :, 1:] = mask_token_id

    expanded_hidden = hidden_states.unsqueeze(2).expand(
        bsz, seq_len, depth, hidden_states.shape[-1]
    )

    if position_ids is None:
        base_pos = torch.arange(seq_len, device=device).view(1, seq_len).expand(bsz, seq_len)
    else:
        base_pos = position_ids.view(bsz, seq_len).long()
    expanded_pos = base_pos.unsqueeze(-1).expand(bsz, seq_len, depth)

    if attention_mask is None:
        attention_mask = torch.ones(bsz, seq_len, device=device, dtype=torch.long)
    expanded_attention_mask = attention_mask.unsqueeze(-1).expand(bsz, seq_len, depth)

    target_pos = torch.arange(seq_len, device=device).view(1, seq_len, 1) + depth_ids + 1
    expanded_loss = loss_mask.unsqueeze(-1).bool() & (target_pos < seq_len)
    expanded_loss = expanded_loss & expanded_attention_mask.bool()

    return PEagleInputs(
        input_ids=expanded_input_ids.reshape(bsz, seq_len * depth),
        hidden_states=expanded_hidden.reshape(bsz, seq_len * depth, hidden_states.shape[-1]),
        loss_mask=expanded_loss.reshape(bsz, seq_len * depth).to(loss_mask.dtype),
        attention_mask=expanded_attention_mask.reshape(bsz, seq_len * depth),
        position_ids=expanded_pos.reshape(bsz, seq_len * depth),
        depth_ids=depth_ids.reshape(bsz, seq_len * depth),
    )


def build_p_eagle_target_hidden_states(
    target_hidden_states: torch.Tensor,
    depth: int,
) -> torch.Tensor:
    bsz, seq_len, hidden_size = target_hidden_states.shape
    padded = F.pad(target_hidden_states, (0, 0, 0, depth), value=0.0)
    pieces = [padded[:, d + 1 : d + 1 + seq_len, :] for d in range(depth)]
    return torch.stack(pieces, dim=2).reshape(bsz, seq_len * depth, hidden_size)


def build_p_eagle_lazy_target(
    target_hidden_states: torch.Tensor,
    target_lm_head_weight: torch.Tensor,
    depth: int,
) -> LazyTarget:
    return LazyTarget(
        hidden_states_padded=build_p_eagle_target_hidden_states(target_hidden_states, depth),
        lm_head_weight=target_lm_head_weight.detach(),
    )


@torch.no_grad()
def build_p_eagle_precomputed_target(
    target_hidden_states: torch.Tensor,
    target_lm_head_weight: torch.Tensor,
    t2d: torch.Tensor,
    loss_mask: torch.Tensor,
    depth: int,
    chunk_size: int = 4096,
) -> PrecomputedTarget:
    target_lm_head_weight = target_lm_head_weight.detach()
    pruned_weight = target_lm_head_weight[t2d]
    expanded_hs = build_p_eagle_target_hidden_states(target_hidden_states, depth)

    bsz, expanded_len, hidden_size = expanded_hs.shape
    loss_mask_bool = loss_mask.bool()
    valid_flat_idx = loss_mask_bool.reshape(-1).nonzero(as_tuple=True)[0]
    valid_hs = expanded_hs.reshape(-1, hidden_size)[valid_flat_idx]

    position_mask_flat = torch.zeros(
        bsz * expanded_len,
        device=expanded_hs.device,
        dtype=torch.float,
    )
    for i in range(0, valid_hs.shape[0], chunk_size):
        chunk_hs = valid_hs[i : i + chunk_size]
        chunk_argmax = F.linear(chunk_hs, target_lm_head_weight).argmax(-1)
        in_draft = t2d[chunk_argmax]
        position_mask_flat[valid_flat_idx[i : i + chunk_size]] = in_draft.float()
    position_mask = position_mask_flat.reshape(bsz, expanded_len)

    target_logits_pruned = F.linear(expanded_hs, pruned_weight)
    target_p = F.softmax(target_logits_pruned.float(), dim=-1)
    return PrecomputedTarget(target_p, position_mask)


def build_p_eagle_sdpa_attention_mask(
    attention_mask: torch.Tensor,
    depth: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    bsz, expanded_len = attention_mask.shape
    device = attention_mask.device
    idx = torch.arange(expanded_len, device=device)
    q_pos = idx.view(expanded_len, 1) // depth
    q_depth = idx.view(expanded_len, 1) % depth
    kv_pos = idx.view(1, expanded_len) // depth
    kv_depth = idx.view(1, expanded_len) % depth

    prefix_context = (kv_pos < q_pos) & (kv_depth == 0)
    same_position = (kv_pos == q_pos) & (kv_depth <= q_depth)
    base = prefix_context | same_position

    query_valid = attention_mask.bool().view(bsz, expanded_len, 1)
    key_valid = attention_mask.bool().view(bsz, 1, expanded_len)
    allowed = base.view(1, expanded_len, expanded_len) & query_valid & key_valid

    additive_mask = torch.zeros((bsz, 1, expanded_len, expanded_len), device=device, dtype=dtype)
    additive_mask = additive_mask.masked_fill(~allowed.unsqueeze(1), torch.finfo(dtype).min)
    return additive_mask


class PEagleModel(Eagle3Model):
    def __init__(
        self,
        draft_model,
        depth: int = 7,
        attention_backend: str = "sdpa",
        gradient_checkpointing: bool = False,
        mask_token_id: int = 0,
        chunk_size: int = 0,
    ):
        super().__init__(
            draft_model=draft_model,
            length=depth,
            attention_backend=attention_backend,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.depth = depth
        self.mask_token_id = int(mask_token_id)
        self.chunk_size = int(chunk_size)
        self.draft_model.config.draft_training_mode = "p_eagle"
        self.draft_model.config.p_eagle_depth = depth
        self.draft_model.config.p_eagle_mask_token_id = self.mask_token_id
        self.draft_model.config.ptd_token_id = self.mask_token_id

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target: Union[PrecomputedTarget, LazyTarget],
        loss_mask: torch.Tensor,
        hidden_states: torch.Tensor,
        past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        if past_key_values is not None:
            raise NotImplementedError("P-EAGLE training does not support past_key_values")
        if self.attention_backend in {"usp", "fa4"}:
            raise NotImplementedError(
                f"P-EAGLE training does not support attention_backend={self.attention_backend}"
            )

        norm_weight, lm_head_weight, norm_eps = self.draft_model.get_lm_head_params()
        projected_hidden_states = self.draft_model.project_hidden_states(hidden_states)

        expanded = build_p_eagle_inputs(
            input_ids=input_ids.clamp(min=0, max=self.draft_model.target_vocab_size - 1),
            hidden_states=projected_hidden_states,
            loss_mask=loss_mask,
            depth=self.depth,
            mask_token_id=self.mask_token_id,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

        inputs_embeds = self.draft_model.embed_input_ids(expanded.input_ids)
        inputs_embeds = inputs_embeds.to(expanded.hidden_states.dtype)

        if self.attention_backend == "sdpa":
            step_attention_mask = build_p_eagle_sdpa_attention_mask(
                expanded.attention_mask,
                depth=self.depth,
                dtype=expanded.hidden_states.dtype,
            )
        else:
            step_attention_mask = expanded.attention_mask

        if self.gradient_checkpointing and self.training:
            hidden_states_out, _, _ = torch_checkpoint(
                self.draft_model.backbone,
                inputs_embeds,
                expanded.hidden_states,
                step_attention_mask,
                expanded.position_ids,
                None,
                None,
                False,
                use_reentrant=False,
            )
        else:
            hidden_states_out, _, _ = self.draft_model.backbone(
                input_embeds=inputs_embeds,
                hidden_states=expanded.hidden_states,
                attention_mask=step_attention_mask,
                position_ids=expanded.position_ids,
                cache_keys=None,
                cache_values=None,
                use_cache=False,
            )

        if self.draft_model.norm_output:
            hidden_states_out = self.draft_model.norm(hidden_states_out)

        plosses = []
        vlosses = []
        acces = []
        acc_counts = []
        expanded_len = expanded.loss_mask.shape[1]
        for depth_idx in range(self.depth):
            depth_mask = expanded.loss_mask * (expanded.depth_ids == depth_idx).to(
                expanded.loss_mask.dtype
            )
            local_sum_loss, local_correct, local_count = self._calculate_loss(
                hidden_states=hidden_states_out,
                target=target,
                mask=depth_mask,
                idx=0,
                seq_length=expanded_len,
                norm_weight=norm_weight,
                lm_head_weight=lm_head_weight,
                norm_eps=norm_eps,
            )
            loss = local_sum_loss / local_count.clamp_min(1.0)
            metric_acc = (
                (local_correct / local_count.clamp_min(1.0)).detach()
                if float(local_count.detach().float().cpu()) > 0.0
                else local_correct.detach().float() * 0.0
            )
            plosses.append(loss)
            vlosses.append(loss.detach())
            acces.append(metric_acc)
            acc_counts.append(local_count.detach().float().to(device=loss.device))

        return plosses, vlosses, acces, acc_counts
