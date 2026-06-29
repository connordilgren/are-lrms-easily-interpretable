# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from collections import namedtuple
from typing import Dict, List, Optional
from transformers.models.gpt2 import GPT2LMHeadModel

# Extended Outputs to include pass_logits for ablation analysis
Outputs = namedtuple("Outputs", ["loss", "inputs_embeds", "logits", "pass_logits"])
MAX_N_LATENT = 8


class AttentionAblationManager:
    """
    Manages attention ablation via forward hooks on GPT2Attention layers.

    This works by intercepting the attention computation and zeroing out
    specific attention weights before they're applied to values.
    """

    def __init__(self):
        self.target_query_idx = None  # Query position to ablate FROM (in current input)
        self.source_positions = []    # Key positions to ablate TO (in full KV cache)
        self.handles = []
        self.is_active = False

    def activate(self, target_query_idx: int, source_positions: List[int]):
        """Activate ablation for the next forward pass."""
        self.target_query_idx = target_query_idx
        self.source_positions = source_positions
        self.is_active = True

    def deactivate(self):
        """Deactivate ablation."""
        self.is_active = False
        self.target_query_idx = None
        self.source_positions = []

    def hook_fn(self, module, args, kwargs, output):
        """
        Hook that modifies attention output.

        For GPT2Attention with output_attentions=True, output is:
            (attn_output, present, attn_weights)
        Otherwise:
            (attn_output, present)

        We need to modify the attention computation itself, not the output.
        This is tricky because the attention is already computed.

        Alternative: use a pre-hook to modify the attention mask, but GPT2
        doesn't support custom 4D masks directly.

        Best solution: Modify the _attn method output by accessing attn_weights
        if available, or by modifying the hidden_states output to simulate
        zeroing out attention to certain keys.
        """
        if not self.is_active:
            return output
        # Can't easily modify attention weights in eager mode without access
        # to intermediate computations. Return output unchanged.
        return output

    def register(self, model):
        """Register hooks on attention layers."""
        if hasattr(model, 'transformer'):
            for layer in model.transformer.h:
                handle = layer.attn.register_forward_hook(
                    self.hook_fn,
                    with_kwargs=True
                )
                self.handles.append(handle)

    def remove(self):
        """Remove all hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles = []


class Coconut(nn.Module):

    def __init__(
        self,
        base_causallm,
        latent_token_id,
        start_latent_id,
        end_latent_id,
        eos_token_id,
        eot_token_id=None,
    ):

        super(Coconut, self).__init__()
        self.gen_forward_cnt = 0
        self.base_causallm = base_causallm
        self.latent_token_id = latent_token_id
        self.eos_token_id = eos_token_id
        self.eot_token_id = eot_token_id  # Llama uses <|eot_id|> for end-of-turn
        self.start_latent_id = start_latent_id
        self.end_latent_id = end_latent_id

        # tested with GPT2 and Llama3
        if isinstance(self.base_causallm, GPT2LMHeadModel):
            self.embedding = self.base_causallm.transformer.get_input_embeddings()
        else:
            self.embedding = self.base_causallm.get_input_embeddings()

    def _patch_attention_for_ablation(self, target_query_idx: int, source_positions: List[int]):
        """
        Temporarily patch the model's attention to ablate specific paths.

        This patches the _attn method of each attention layer to zero out
        attention weights from target_query_idx to source_positions and
        recompute the attention output.
        """
        original_attns = []

        for layer in self.base_causallm.transformer.h:
            original_attn = layer.attn._attn
            original_attns.append(original_attn)

            def make_patched_attn(orig_attn, target_idx, src_positions):
                def patched_attn(query, key, value, attention_mask=None, head_mask=None):
                    # Call original attention to get initial weights
                    attn_output, attn_weights = orig_attn(query, key, value, attention_mask, head_mask)

                    # Modify attention weights if needed
                    # attn_weights shape: [batch, heads, query_len, key_len]
                    if target_idx is not None and 0 <= target_idx < attn_weights.shape[2]:
                        # Clone weights to avoid modifying the original
                        modified_weights = attn_weights.clone()

                        # Zero out attention to source positions
                        for src_pos in src_positions:
                            if 0 <= src_pos < modified_weights.shape[3]:
                                modified_weights[:, :, target_idx, src_pos] = 0

                        # Renormalize the row (so attention weights sum to 1)
                        row_sum = modified_weights[:, :, target_idx, :].sum(dim=-1, keepdim=True)
                        row_sum = torch.clamp(row_sum, min=1e-9)  # Avoid division by zero
                        modified_weights[:, :, target_idx, :] = modified_weights[:, :, target_idx, :] / row_sum

                        # Recompute attention output with modified weights
                        # Only need to recompute for the target row
                        # attn_output shape: [batch, heads, query_len, head_dim]
                        # value shape: [batch, heads, key_len, head_dim]
                        modified_output = attn_output.clone()
                        # Compute: modified_output[:, :, target_idx, :] = modified_weights[:, :, target_idx, :] @ value
                        modified_output[:, :, target_idx, :] = torch.matmul(
                            modified_weights[:, :, target_idx:target_idx+1, :],  # [batch, heads, 1, key_len]
                            value  # [batch, heads, key_len, head_dim]
                        ).squeeze(2)  # [batch, heads, head_dim]

                        return modified_output, modified_weights

                    return attn_output, attn_weights

                return patched_attn

            layer.attn._attn = make_patched_attn(original_attn, target_query_idx, source_positions)

        return original_attns

    def _restore_attention(self, original_attns):
        """Restore original attention methods."""
        for layer, orig_attn in zip(self.base_causallm.transformer.h, original_attns):
            layer.attn._attn = orig_attn

    def _patch_attention_total_ablation(self, source_positions: List[int]):
        """
        Patch attention to block ALL positions from attending to source positions.

        Unlike _patch_attention_for_ablation which only blocks ONE target position,
        this blocks attention to source_positions from ALL query positions.
        This captures indirect/multi-hop attention effects.

        Args:
            source_positions: Key positions to block attention TO

        Returns:
            List of original _attn methods to restore later
        """
        original_attns = []

        for layer in self.base_causallm.transformer.h:
            original_attn = layer.attn._attn
            original_attns.append(original_attn)

            def make_patched_attn(orig_attn, src_positions):
                def patched_attn(query, key, value, attention_mask=None, head_mask=None):
                    # Call original attention to get initial weights
                    attn_output, attn_weights = orig_attn(query, key, value, attention_mask, head_mask)

                    # attn_weights shape: [batch, heads, query_len, key_len]
                    modified_weights = attn_weights.clone()

                    # Zero out attention to source positions for ALL query positions
                    for src_pos in src_positions:
                        if 0 <= src_pos < modified_weights.shape[3]:
                            modified_weights[:, :, :, src_pos] = 0

                    # Renormalize each row (so attention weights sum to 1)
                    row_sum = modified_weights.sum(dim=-1, keepdim=True)
                    row_sum = torch.clamp(row_sum, min=1e-9)
                    modified_weights = modified_weights / row_sum

                    # Recompute attention output with modified weights
                    modified_output = torch.matmul(modified_weights, value)

                    return modified_output, modified_weights

                return patched_attn

            layer.attn._attn = make_patched_attn(original_attn, source_positions)

        return original_attns

    def _patch_attention_total_ablation_single_layer(
        self,
        layer_idx: int,
        source_positions: List[int]
    ):
        """
        Patch attention for a SINGLE layer to block ALL positions from attending to sources.

        Unlike _patch_attention_for_single_layer_ablation which only blocks ONE target position,
        this blocks attention to source_positions from ALL query positions at a specific layer.
        This captures indirect/multi-hop attention effects at that layer.

        Args:
            layer_idx: Which transformer layer to patch (0-indexed)
            source_positions: Key positions to block attention TO

        Returns:
            Original _attn method to restore later
        """
        layer = self.base_causallm.transformer.h[layer_idx]
        original_attn = layer.attn._attn

        def make_patched_attn(orig_attn, src_positions):
            def patched_attn(query, key, value, attention_mask=None, head_mask=None):
                # Call original attention to get initial weights
                attn_output, attn_weights = orig_attn(query, key, value, attention_mask, head_mask)

                # attn_weights shape: [batch, heads, query_len, key_len]
                modified_weights = attn_weights.clone()

                # Zero out attention to source positions for ALL query positions
                for src_pos in src_positions:
                    if 0 <= src_pos < modified_weights.shape[3]:
                        modified_weights[:, :, :, src_pos] = 0

                # Renormalize each row (so attention weights sum to 1)
                row_sum = modified_weights.sum(dim=-1, keepdim=True)
                row_sum = torch.clamp(row_sum, min=1e-9)
                modified_weights = modified_weights / row_sum

                # Recompute attention output with modified weights
                modified_output = torch.matmul(modified_weights, value)

                return modified_output, modified_weights

            return patched_attn

        layer.attn._attn = make_patched_attn(original_attn, source_positions)
        return original_attn

    def _patch_attention_for_single_layer_ablation(
        self,
        layer_idx: int,
        target_query_idx: int,
        source_positions: List[int]
    ):
        """
        Patch attention for a SINGLE layer only (not all layers).

        This enables per-layer ablation analysis to understand which layer's
        attention to a particular position is most important.

        Args:
            layer_idx: Which transformer layer to patch (0-indexed)
            target_query_idx: Query position to ablate FROM (in current input chunk)
            source_positions: Key positions to ablate TO (in full KV cache)

        Returns:
            Original _attn method to restore later
        """
        layer = self.base_causallm.transformer.h[layer_idx]
        original_attn = layer.attn._attn

        def make_patched_attn(orig_attn, target_idx, src_positions):
            def patched_attn(query, key, value, attention_mask=None, head_mask=None):
                # Call original attention to get initial weights
                attn_output, attn_weights = orig_attn(query, key, value, attention_mask, head_mask)

                # Modify attention weights if needed
                # attn_weights shape: [batch, heads, query_len, key_len]
                if target_idx is not None and 0 <= target_idx < attn_weights.shape[2]:
                    # Clone weights to avoid modifying the original
                    modified_weights = attn_weights.clone()

                    # Zero out attention to source positions
                    for src_pos in src_positions:
                        if 0 <= src_pos < modified_weights.shape[3]:
                            modified_weights[:, :, target_idx, src_pos] = 0

                    # Renormalize the row (so attention weights sum to 1)
                    row_sum = modified_weights[:, :, target_idx, :].sum(dim=-1, keepdim=True)
                    row_sum = torch.clamp(row_sum, min=1e-9)
                    modified_weights[:, :, target_idx, :] = modified_weights[:, :, target_idx, :] / row_sum

                    # Recompute attention output with modified weights
                    modified_output = attn_output.clone()
                    modified_output[:, :, target_idx, :] = torch.matmul(
                        modified_weights[:, :, target_idx:target_idx+1, :],
                        value
                    ).squeeze(2)

                    return modified_output, modified_weights

                return attn_output, attn_weights

            return patched_attn

        layer.attn._attn = make_patched_attn(original_attn, target_query_idx, source_positions)
        return original_attn

    def _restore_single_layer_attention(self, layer_idx: int, original_attn):
        """Restore original attention method for a single layer."""
        self.base_causallm.transformer.h[layer_idx].attn._attn = original_attn

    def _patch_attention_for_token_groups(
        self,
        token_groups: List[Dict],
        target_query_idx: int
    ) -> List:
        """
        Patch attention to ablate token groups with layer-specific ranges.

        Each token group specifies:
        - 'positions': List[int] - positions to ablate
        - 'layer_range': Tuple[int, int] - (start, end) layers to ablate at (exclusive end)

        For each layer, we collect all positions that should be ablated at that layer
        by checking which token groups include this layer in their range.

        Args:
            token_groups: List of dicts, each with 'positions' and 'layer_range'
            target_query_idx: Query position to ablate FROM (in current input chunk)

        Returns:
            List of original _attn methods (one per layer) to restore later
        """
        original_attns = []
        num_layers = len(self.base_causallm.transformer.h)

        for layer_idx, layer in enumerate(self.base_causallm.transformer.h):
            original_attn = layer.attn._attn
            original_attns.append(original_attn)

            # Collect all positions to ablate at this layer
            positions_to_ablate = set()
            for group in token_groups:
                layer_start, layer_end = group['layer_range']
                if layer_start <= layer_idx < layer_end:
                    positions_to_ablate.update(group['positions'])

            if positions_to_ablate:
                # Create patched attention that zeros these positions
                def make_patched_attn(orig_attn, target_idx, src_positions):
                    def patched_attn(query, key, value, attention_mask=None, head_mask=None):
                        # Call original attention to get initial weights
                        attn_output, attn_weights = orig_attn(query, key, value, attention_mask, head_mask)

                        # Modify attention weights if needed
                        if target_idx is not None and 0 <= target_idx < attn_weights.shape[2]:
                            modified_weights = attn_weights.clone()

                            # Zero out attention to source positions
                            for src_pos in src_positions:
                                if 0 <= src_pos < modified_weights.shape[3]:
                                    modified_weights[:, :, target_idx, src_pos] = 0

                            # Renormalize the row
                            row_sum = modified_weights[:, :, target_idx, :].sum(dim=-1, keepdim=True)
                            row_sum = torch.clamp(row_sum, min=1e-9)
                            modified_weights[:, :, target_idx, :] = modified_weights[:, :, target_idx, :] / row_sum

                            # Recompute attention output for the target row
                            modified_output = attn_output.clone()
                            modified_output[:, :, target_idx, :] = torch.matmul(
                                modified_weights[:, :, target_idx:target_idx+1, :],
                                value
                            ).squeeze(2)

                            return modified_output, modified_weights

                        return attn_output, attn_weights

                    return patched_attn

                layer.attn._attn = make_patched_attn(original_attn, target_query_idx, list(positions_to_ablate))

        return original_attns

    def _run_forward_with_ablation(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values=None,
        target_position: int = None,
        source_positions: List[int] = None,
        ablation_type: str = 'direct',
        target_layer: int = None,
        token_groups: List[Dict] = None
    ):
        """
        Run forward pass with optional attention ablation.

        For ablation, we patch the attention _attn method to zero out
        attention weights from target to sources after softmax, then renormalize.

        Args:
            ablation_type: 'direct' (only target position) or 'total' (all positions)
            target_layer: If specified, only ablate this specific layer (0-indexed).
                          If None, ablate all layers (original behavior).
            token_groups: List of dicts for token group ablation. Each dict has:
                          - 'positions': List[int] - positions to ablate
                          - 'layer_range': Tuple[int, int] - (start, end) layers
                          When provided, this takes precedence over other ablation configs.
        """
        # Token group ablation (takes precedence over other ablation types)
        if token_groups and len(token_groups) > 0:
            # Convert target_position to query index
            query_len = inputs_embeds.shape[1]
            if past_key_values is not None:
                kv_len = past_key_values[0][0].shape[2]
                target_query_idx = target_position - kv_len if target_position is not None else None
            else:
                target_query_idx = target_position

            if target_query_idx is not None and 0 <= target_query_idx < query_len:
                original_attns = self._patch_attention_for_token_groups(token_groups, target_query_idx)
                try:
                    outputs = self.base_causallm(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        output_hidden_states=True,
                        output_attentions=True,
                        use_cache=True
                    )
                finally:
                    self._restore_attention(original_attns)
                return outputs

        if source_positions and len(source_positions) > 0:
            if ablation_type == 'total':
                # Total ablation: block ALL positions from attending to sources
                if target_layer is not None:
                    # Single-layer total ablation
                    num_layers = len(self.base_causallm.transformer.h)
                    if 0 <= target_layer < num_layers:
                        original_attn = self._patch_attention_total_ablation_single_layer(
                            target_layer, source_positions
                        )
                        try:
                            outputs = self.base_causallm(
                                inputs_embeds=inputs_embeds,
                                attention_mask=attention_mask,
                                position_ids=position_ids,
                                past_key_values=past_key_values,
                                output_hidden_states=True,
                                output_attentions=True,
                                use_cache=True
                            )
                        finally:
                            self._restore_single_layer_attention(target_layer, original_attn)
                        return outputs
                else:
                    # All-layer total ablation
                    original_attns = self._patch_attention_total_ablation(source_positions)
                    try:
                        outputs = self.base_causallm(
                            inputs_embeds=inputs_embeds,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            past_key_values=past_key_values,
                            output_hidden_states=True,
                            output_attentions=True,
                            use_cache=True
                        )
                    finally:
                        self._restore_attention(original_attns)
                    return outputs
            elif target_position is not None:
                # Direct ablation: only block from target position
                # Convert target_position (in full sequence) to query index (in current input)
                query_len = inputs_embeds.shape[1]
                if past_key_values is not None:
                    kv_len = past_key_values[0][0].shape[2]
                    target_query_idx = target_position - kv_len
                else:
                    target_query_idx = target_position

                # Only patch if target is in current query range
                if 0 <= target_query_idx < query_len:
                    if target_layer is not None:
                        # Single-layer ablation
                        num_layers = len(self.base_causallm.transformer.h)
                        if 0 <= target_layer < num_layers:
                            original_attn = self._patch_attention_for_single_layer_ablation(
                                target_layer, target_query_idx, source_positions
                            )
                            try:
                                outputs = self.base_causallm(
                                    inputs_embeds=inputs_embeds,
                                    attention_mask=attention_mask,
                                    position_ids=position_ids,
                                    past_key_values=past_key_values,
                                    output_hidden_states=True,
                                    output_attentions=True,
                                    use_cache=True
                                )
                            finally:
                                self._restore_single_layer_attention(target_layer, original_attn)
                            return outputs
                    else:
                        # All-layer ablation (original behavior)
                        original_attns = self._patch_attention_for_ablation(target_query_idx, source_positions)
                        try:
                            outputs = self.base_causallm(
                                inputs_embeds=inputs_embeds,
                                attention_mask=attention_mask,
                                position_ids=position_ids,
                                past_key_values=past_key_values,
                                output_hidden_states=True,
                                output_attentions=True,  # Need this for _attn to return weights
                                use_cache=True
                            )
                        finally:
                            self._restore_attention(original_attns)
                        return outputs

        # No ablation or target not in range
        return self.base_causallm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_hidden_states=True,
            use_cache=True
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        labels,
        position_ids,
        ablation_config: Optional[Dict] = None,
        **kwargs
    ):
        """
        Forward pass with optional ablation support for importance analysis.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            labels: Labels for loss computation
            position_ids: Position IDs
            ablation_config: Optional dict with:
                - 'target_pass': int or 'final' - which pass to ablate (0-indexed, or 'final' for final pass)
                - 'target_position_in_chunk': int - explicit position within the chunk to target (for final pass)
                - 'source_positions': List[int] - which positions to mask attention to
                - 'collect_pass_logits': bool - whether to collect logits at each pass
                - 'ablation_type': str - 'direct' (default) or 'total'. Direct ablation only blocks
                                         from target position. Total ablation blocks ALL positions.
                - 'token_groups': List[Dict] - for split ablation, each dict has 'positions' and 'layer_range'
            **kwargs: Additional arguments (ignored)

        Returns:
            Outputs namedtuple with loss, inputs_embeds, logits, and optionally pass_logits
        """
        logits = []
        pass_logits = [] if ablation_config and ablation_config.get('collect_pass_logits') else None

        latent_indices = (
            input_ids == self.latent_token_id
        ).nonzero()

        latent_lists = [
            [idx[1].item() for idx in latent_indices if idx[0] == i]
            for i in range(input_ids.shape[0])
        ]

        max_n_latents = max([len(l) for l in latent_lists]) if latent_lists else 0

        next_compute_range = (0, input_ids.shape[1])
        inputs_embeds = self.embedding(input_ids)

        if max_n_latents > 0:
            next_compute_range = (0, latent_indices[:, 1].min().item())

        kv_cache = None
        device = input_ids.device

        # Ablation configuration
        target_pass = ablation_config.get('target_pass') if ablation_config else None
        target_position_in_chunk = ablation_config.get('target_position_in_chunk') if ablation_config else None
        source_positions = ablation_config.get('source_positions', []) if ablation_config else []
        ablation_type = ablation_config.get('ablation_type', 'direct') if ablation_config else 'direct'
        target_layer = ablation_config.get('target_layer') if ablation_config else None
        token_groups = ablation_config.get('token_groups', []) if ablation_config else []

        for pass_idx in range(max_n_latents):
            # Check if we should apply ablation for this pass
            # Token groups take precedence over source_positions
            has_ablation_config = len(token_groups) > 0 or len(source_positions) > 0
            apply_ablation = (
                ablation_config is not None and
                target_pass == pass_idx and
                has_ablation_config
            )

            # Target position is the last position in this pass (produces latent)
            target_position = next_compute_range[1] - 1 if apply_ablation else None

            if kv_cache is None:
                outputs = self._run_forward_with_ablation(
                    inputs_embeds=inputs_embeds[:, next_compute_range[0]:next_compute_range[1], :],
                    attention_mask=attention_mask[:, next_compute_range[0]:next_compute_range[1]],
                    position_ids=position_ids[:, next_compute_range[0]:next_compute_range[1]],
                    past_key_values=None,
                    target_position=target_position,
                    source_positions=source_positions if apply_ablation else None,
                    ablation_type=ablation_type if apply_ablation else 'direct',
                    target_layer=target_layer if apply_ablation else None,
                    token_groups=token_groups if apply_ablation else None
                )
                hidden_states_offset = 0
            else:
                past_key_values = [
                    (k[:, :, :next_compute_range[0], :], v[:, :, :next_compute_range[0], :])
                    for k, v in kv_cache
                ]

                outputs = self._run_forward_with_ablation(
                    inputs_embeds=inputs_embeds[:, next_compute_range[0]:next_compute_range[1], :],
                    attention_mask=attention_mask[:, :next_compute_range[1]],
                    position_ids=position_ids[:, next_compute_range[0]:next_compute_range[1]],
                    past_key_values=past_key_values,
                    target_position=target_position,
                    source_positions=source_positions if apply_ablation else None,
                    ablation_type=ablation_type if apply_ablation else 'direct',
                    target_layer=target_layer if apply_ablation else None,
                    token_groups=token_groups if apply_ablation else None
                )
                hidden_states_offset = next_compute_range[0]

            logits.append(outputs.logits)

            if pass_logits is not None:
                pass_logits.append(outputs.logits.clone())

            next_compute_range = (
                next_compute_range[1],
                input_ids.shape[1] if pass_idx + 1 >= max_n_latents else next_compute_range[1] + 1
            )

            hidden_states = outputs.hidden_states[-1]
            kv_cache = outputs.past_key_values

            filling_indices = [
                (instance_idx, mask_list[pass_idx])
                for instance_idx, mask_list in enumerate(latent_lists)
                if len(mask_list) > pass_idx
            ]

            tensor_list = [
                [inputs_embeds[batch_idx, pos, :] for pos in range(inputs_embeds.shape[1])]
                for batch_idx in range(inputs_embeds.shape[0])
            ]

            for idx_pair in filling_indices:
                batch_idx, token_idx = idx_pair
                tensor_list[batch_idx][token_idx] = hidden_states[
                    batch_idx, token_idx - 1 - hidden_states_offset, :
                ]

            inputs_embeds = torch.stack([
                torch.stack(tensor_list[batch_idx])
                for batch_idx in range(inputs_embeds.shape[0])
            ])

        # Final pass (with optional ablation support)
        # Check if ablation targets the final pass
        has_ablation_config = len(token_groups) > 0 or len(source_positions) > 0
        apply_final_ablation = (
            ablation_config is not None and
            (target_pass == 'final' or target_pass == max_n_latents) and
            has_ablation_config
        )

        if apply_final_ablation:
            # Determine target position within the final chunk
            chunk_start = next_compute_range[0]
            chunk_end = next_compute_range[1]
            chunk_len = chunk_end - chunk_start

            if target_position_in_chunk is not None:
                # Use explicit position within chunk
                target_query_idx = target_position_in_chunk
            else:
                # Default to first position (last latent predicting end-latent)
                target_query_idx = 0

            # Ensure target is within chunk
            if 0 <= target_query_idx < chunk_len:
                # Convert to full sequence position for ablation
                target_position = chunk_start + target_query_idx

                past_key_values = (
                    [(k[:, :, :chunk_start, :], v[:, :, :chunk_start, :]) for k, v in kv_cache]
                    if kv_cache else None
                )

                outputs = self._run_forward_with_ablation(
                    inputs_embeds=inputs_embeds[:, chunk_start:chunk_end, :],
                    attention_mask=attention_mask[:, :chunk_end],
                    position_ids=position_ids[:, chunk_start:chunk_end],
                    past_key_values=past_key_values,
                    target_position=target_position,
                    source_positions=source_positions,
                    ablation_type=ablation_type,
                    target_layer=target_layer,
                    token_groups=token_groups
                )
            else:
                # Target out of range, run without ablation
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[:, next_compute_range[0]:next_compute_range[1], :],
                    attention_mask=attention_mask[:, :next_compute_range[1]],
                    position_ids=position_ids[:, next_compute_range[0]:next_compute_range[1]],
                    past_key_values=(
                        [(k[:, :, :next_compute_range[0], :], v[:, :, :next_compute_range[0], :]) for k, v in kv_cache]
                        if kv_cache else None
                    ),
                    output_hidden_states=True,
                )
        else:
            # No ablation on final pass
            outputs = self.base_causallm(
                inputs_embeds=inputs_embeds[:, next_compute_range[0]:next_compute_range[1], :],
                attention_mask=attention_mask[:, :next_compute_range[1]],
                position_ids=position_ids[:, next_compute_range[0]:next_compute_range[1]],
                past_key_values=(
                    [(k[:, :, :next_compute_range[0], :], v[:, :, :next_compute_range[0], :]) for k, v in kv_cache]
                    if kv_cache else None
                ),
                output_hidden_states=True,
            )

        logits.append(outputs.logits)
        if pass_logits is not None:
            pass_logits.append(outputs.logits.clone())

        self.gen_forward_cnt += max_n_latents + 1

        logits = torch.cat(logits, dim=-2)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return Outputs(loss=loss, inputs_embeds=inputs_embeds, logits=logits, pass_logits=pass_logits)

    def train(self):
        self.base_causallm.train()

    def eval(self):
        self.base_causallm.eval()

    def generate(
        self,
        input_ids,
        attention_mask,
        max_new_tokens=16,
        output_embedding=False,
        synced_gpus=False,
        **kwargs
    ):
        self.gen_forward_cnt = 0
        assert input_ids.shape[0] == 1, "only support batch_size == 1 now"

        tokens = input_ids[0].detach().tolist()
        labels = input_ids.clone()

        outputs = self.forward(
            input_ids,
            torch.ones_like(input_ids, device=input_ids.device),
            labels,
            torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device).reshape(1, -1),
        )
        inputs_embeds = outputs.inputs_embeds

        next_token = torch.argmax(outputs.logits[0, -1]).item()
        tokens.append(next_token)
        new_token_embed = self.embedding(torch.tensor(next_token, device=input_ids.device)).view(1, 1, -1)
        new_inputs_embeds = torch.cat((inputs_embeds, new_token_embed), dim=1)

        for _ in range(max_new_tokens - 1):
            outputs = self.base_causallm(inputs_embeds=new_inputs_embeds)
            self.gen_forward_cnt += 1
            next_token = torch.argmax(outputs.logits[0, -1]).item()
            if next_token == self.eos_token_id:
                break
            if self.eot_token_id is not None and next_token == self.eot_token_id:
                break
            tokens.append(next_token)
            new_token_embed = self.embedding(torch.tensor(next_token, device=input_ids.device)).view(1, 1, -1)
            new_inputs_embeds = torch.cat((new_inputs_embeds, new_token_embed), dim=1)

        if synced_gpus:
            while self.gen_forward_cnt < max_new_tokens + MAX_N_LATENT:
                self.gen_forward_cnt += 1
                _ = self.base_causallm(inputs_embeds=new_inputs_embeds)

        if output_embedding:
            return torch.tensor(tokens).view(1, -1), new_inputs_embeds
        else:
            return torch.tensor(tokens).view(1, -1)
