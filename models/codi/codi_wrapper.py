"""
CODI (Chain-of-Thought Distillation into Continuous Space) inference wrapper.

This is a simplified, inference-only version of CODI extracted from:
/Users/connordilgren/programming/codi/src/model.py

Removes all training-specific code (LoRA, losses, distillation) and keeps
only what's needed for generation.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, NamedTuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType


class GenerateResult(NamedTuple):
    """Result from CODI generate with optional ablation outputs."""
    output_ids: torch.Tensor  # Generated token IDs
    iteration_logits: Optional[List[torch.Tensor]] = None  # Logits from each iteration


class EncodedState(NamedTuple):
    """Cached state after encoding question, for efficient ablation analysis."""
    past_key_values: List[tuple]  # KV cache from question encoding
    initial_latent_embd: torch.Tensor  # Initial latent embedding [batch, 1, hidden_dim]


class LatentIterationResult(NamedTuple):
    """Result from running latent iterations only (for ablation analysis)."""
    iteration_logits: List[torch.Tensor]  # Logits from each iteration


def ablate_kv_cache_positions(
    past_key_values: List[tuple],
    source_positions: List[int]
) -> List[tuple]:
    """
    Zero out values at specified positions in KV cache for ablation.

    This is a clean way to ablate attention that works with SDPA:
    - SDPA computes: output = softmax(Q @ K.T / sqrt(d)) @ V
    - If V[:, :, pos, :] = 0, the contribution from position pos is zero
    - This effectively "removes" that position from the model's view

    Args:
        past_key_values: List of (key, value) tuples per layer
            Each tensor has shape [batch, num_heads, seq_len, head_dim]
        source_positions: Positions to ablate (zero out values)

    Returns:
        Modified KV cache with values zeroed at source_positions
    """
    if not source_positions:
        return past_key_values

    modified_kv = []
    for layer_k, layer_v in past_key_values:
        # Clone value tensor to avoid modifying original
        new_v = layer_v.clone()
        for pos in source_positions:
            if pos < new_v.shape[2]:  # Check position is valid
                new_v[:, :, pos, :] = 0
        modified_kv.append((layer_k, new_v))
    return modified_kv


def patch_attention_for_ablation(
    model,
    target_query_idx: int,
    source_positions: List[int],
    target_layer: Optional[int] = None
):
    """
    Patch the model's attention layers to ablate specific attention paths.

    This patches the _attn method of each attention layer to zero out
    attention weights from target_query_idx to source_positions and
    recompute the attention output.

    IMPORTANT: Must use output_attentions=True when calling the model
    to force fallback from SDPA to eager attention (which uses _attn).

    Args:
        model: The GPT2 model (or base model from PEFT wrapper)
        target_query_idx: Query position to ablate FROM (in current input chunk)
        source_positions: Key positions to ablate TO (in full KV cache)
        target_layer: If specified, only patch this layer (0-indexed).
                     If None, patch all layers.

    Returns:
        List of original _attn methods to restore later
    """
    original_attns = []

    # Get transformer layers (handle PEFT wrapper)
    if hasattr(model, 'get_base_model'):
        base_model = model.get_base_model()
    else:
        base_model = model

    for layer_idx, layer in enumerate(base_model.transformer.h):
        # Skip layers that aren't the target (if target_layer is specified)
        if target_layer is not None and layer_idx != target_layer:
            continue
        original_attn = layer.attn._attn
        original_attns.append((layer.attn, original_attn))

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
                    modified_output = attn_output.clone()
                    modified_output[:, :, target_idx, :] = torch.matmul(
                        modified_weights[:, :, target_idx:target_idx+1, :],
                        value
                    ).squeeze(2)

                    return modified_output, modified_weights

                return attn_output, attn_weights

            return patched_attn

        layer.attn._attn = make_patched_attn(original_attn, target_query_idx, source_positions)

    return original_attns


def restore_attention(original_attns):
    """Restore original _attn methods after patching."""
    for attn_module, orig_attn in original_attns:
        attn_module._attn = orig_attn


def patch_attention_for_token_groups(
    model,
    token_groups: List[Dict],
    target_query_idx: int = 0
) -> List[tuple]:
    """
    Patch attention with layer-specific ablation for token groups.

    Each token_group dict has:
    - 'positions': List[int] - positions in KV cache to ablate
    - 'layer_range': Tuple[int, int] - (start, end) layers to apply ablation (inclusive start, exclusive end)

    This allows ablating different positions at different layers, which is needed
    for the "split" ablation experiment where latent tokens span upper/lower layer halves.

    Args:
        model: The GPT2 model (or PEFT wrapper)
        token_groups: List of token group specifications
        target_query_idx: Query position to ablate FROM (usually 0 for single-token input)

    Returns:
        List of (attn_module, original_attn) tuples to restore later
    """
    original_attns = []

    if hasattr(model, 'get_base_model'):
        base_model = model.get_base_model()
    else:
        base_model = model

    num_layers = len(base_model.transformer.h)

    for layer_idx, layer in enumerate(base_model.transformer.h):
        original_attn = layer.attn._attn
        original_attns.append((layer.attn, original_attn))

        # Collect positions to ablate at this layer
        positions_to_ablate = set()
        for group in token_groups:
            layer_start, layer_end = group['layer_range']
            if layer_start <= layer_idx < layer_end:
                positions_to_ablate.update(group['positions'])

        if positions_to_ablate:
            # Create patched attention for this layer
            def make_patched_attn(orig_attn, target_idx, src_positions):
                def patched_attn(query, key, value, attention_mask=None, head_mask=None):
                    attn_output, attn_weights = orig_attn(query, key, value, attention_mask, head_mask)

                    if target_idx is not None and 0 <= target_idx < attn_weights.shape[2]:
                        modified_weights = attn_weights.clone()
                        for src_pos in src_positions:
                            if 0 <= src_pos < modified_weights.shape[3]:
                                modified_weights[:, :, target_idx, src_pos] = 0

                        # Renormalize
                        row_sum = modified_weights[:, :, target_idx, :].sum(dim=-1, keepdim=True)
                        row_sum = torch.clamp(row_sum, min=1e-9)
                        modified_weights[:, :, target_idx, :] = modified_weights[:, :, target_idx, :] / row_sum

                        # Recompute output for modified row
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


def patch_attention_total_ablation(model, source_positions: List[int]):
    """
    Patch the model's attention layers to block ALL positions from attending to source positions.

    Unlike patch_attention_for_ablation which only blocks a specific query position,
    this blocks ALL query positions from attending to the source positions.
    This captures indirect/multi-hop attention effects.

    Args:
        model: The GPT2 model (or base model from PEFT wrapper)
        source_positions: Key positions to block attention TO

    Returns:
        List of original _attn methods to restore later
    """
    original_attns = []

    # Get transformer layers (handle PEFT wrapper)
    if hasattr(model, 'get_base_model'):
        base_model = model.get_base_model()
    else:
        base_model = model

    for layer in base_model.transformer.h:
        original_attn = layer.attn._attn
        original_attns.append((layer.attn, original_attn))

        def make_patched_attn(orig_attn, src_positions):
            def patched_attn(query, key, value, attention_mask=None, head_mask=None):
                # Call original attention to get initial weights
                attn_output, attn_weights = orig_attn(query, key, value, attention_mask, head_mask)

                # Modify attention weights: block ALL queries from attending to source positions
                # attn_weights shape: [batch, heads, query_len, key_len]
                modified_weights = attn_weights.clone()

                for src_pos in src_positions:
                    if 0 <= src_pos < modified_weights.shape[3]:
                        modified_weights[:, :, :, src_pos] = 0

                # Renormalize each row (so attention weights sum to 1)
                row_sum = modified_weights.sum(dim=-1, keepdim=True)
                row_sum = torch.clamp(row_sum, min=1e-9)  # Avoid division by zero
                modified_weights = modified_weights / row_sum

                # Recompute attention output with modified weights
                modified_output = torch.matmul(modified_weights, value)

                return modified_output, modified_weights

            return patched_attn

        layer.attn._attn = make_patched_attn(original_attn, source_positions)

    return original_attns


class CODIInference(nn.Module):
    """
    CODI model for inference only.

    CODI uses continuous latent embeddings (not tokens!) to compress
    chain-of-thought reasoning into a continuous representation.
    """

    def __init__(
        self,
        model_name_or_path: str = "gpt2",
        num_latent: int = 6,
        use_prj: bool = True,
        prj_dim: int = None,
        prj_dropout: float = 0.0,
        prj_no_ln: bool = False,
        use_lora: bool = True,
        lora_r: int = 128,
        lora_alpha: int = 32,
        device: str = "cuda"
    ):
        """
        Initialize CODI for inference.

        Args:
            model_name_or_path: Base model path (default: gpt2)
            num_latent: Number of latent iterations
            use_prj: Whether to use projection layer
            prj_dim: Projection layer hidden dimension
            prj_dropout: Projection layer dropout
            prj_no_ln: Whether to skip layer norm in projection
            use_lora: Whether to use LoRA
            lora_r: LoRA rank
            lora_alpha: LoRA alpha
            device: Device to load on
        """
        super().__init__()

        self.model_name = model_name_or_path
        self.num_latent = num_latent
        self.use_prj = use_prj
        self.prj_no_ln = prj_no_ln
        self.device_str = device

        # Load base model
        print(f"Loading base model {model_name_or_path}...")
        self.codi = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            resume_download=True
        )

        # Special tokens
        ori_vocab_size = self.codi.config.vocab_size
        self.pad_token_id = ori_vocab_size
        self.bot_id = ori_vocab_size + 1  # Beginning of thought (also used as latent placeholder)
        self.eot_id = ori_vocab_size + 2  # End of thought

        # Resize for special tokens (must match checkpoint: 3 tokens only)
        self.codi.resize_token_embeddings(ori_vocab_size + 3)

        # Model dimensions
        self.dim = self.codi.config.hidden_size

        # Default prj_dim to model's hidden size if not specified
        if prj_dim is None:
            prj_dim = self.dim

        # Apply LoRA
        if use_lora:
            print(f"Applying LoRA (r={lora_r}, alpha={lora_alpha})...")
            # Determine target modules based on model type
            if "gpt2" in model_name_or_path.lower():
                target_modules = ["c_attn", "c_proj", "c_fc"]
            elif any(name in model_name_or_path.lower() for name in ["llama", "mistral", "falcon", "qwen"]):
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
            elif "phi" in model_name_or_path.lower():
                target_modules = ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"]
            else:
                target_modules = ["c_attn", "c_proj", "c_fc"]  # Default to GPT2-style

            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,  # Set to False to allow loading trained weights
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=0.1,
                target_modules=target_modules,
                init_lora_weights=True
            )
            self.codi = get_peft_model(self.codi, lora_config)
            print("LoRA applied successfully")

        # Tokenizer - load based on model type
        # CODI checkpoints may not include tokenizer files, so we infer from config
        model_type = self.codi.config.model_type
        if model_type == "gpt2":
            tokenizer_name = "gpt2"
        else:
            # For other model types, try loading from model_name
            tokenizer_name = self.model_name

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)

        # CRITICAL: Set padding side to LEFT to match original CODI implementation
        # This ensures consistent absolute positions for positional embeddings
        self.tokenizer.padding_side = "left"

        # Setup special tokens for tokenizer
        # Note: We do NOT add bot/eot/latent to tokenizer vocab because
        # the tokenizer would assign different IDs than the model's fixed IDs
        # Instead, we'll handle decoding of these tokens manually
        special_tokens_dict = {}
        if self.tokenizer.pad_token_id is None:
            special_tokens_dict['pad_token'] = '[PAD]'
            self.tokenizer.add_special_tokens(special_tokens_dict)
            self.tokenizer.pad_token_id = self.pad_token_id

        # For latent visualization, use bot_id as placeholder
        # Decoding will replace bot tokens in latent positions with <|latent|>
        self.latent_id = self.bot_id

        # Get Llama's <|eot_id|> token for stopping generation (different from internal eot_id)
        llama_eot_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if llama_eot_id == self.tokenizer.unk_token_id:
            self.llama_eot_token_id = None  # Token doesn't exist (e.g., GPT-2)
        else:
            self.llama_eot_token_id = llama_eot_id

        # Projection layer (optional)
        # IMPORTANT: Must match original CODI structure exactly for checkpoint loading
        if use_prj:
            self.prj = nn.Sequential(
                nn.Dropout(prj_dropout),
                nn.Linear(self.dim, prj_dim),
                nn.GELU(),
                nn.Linear(prj_dim, self.dim)
            )
            # Add LayerNorm as named module (not indexed) to match checkpoint keys
            if not prj_no_ln:
                self.prj.add_module("ln", nn.LayerNorm(self.dim))

            # Convert to bfloat16 to match model dtype
            self.prj = self.prj.to(torch.bfloat16)

        print("CODI inference model initialized")

    def get_embd(self, model, model_name):
        """
        Get embedding layer from model.

        Handles different model architectures (GPT2, Pythia, Llama, etc.)
        """
        try:
            if "pythia" in model_name:
                return model.get_base_model().gpt_neox.embed_in
            elif "gpt2" in model_name:
                try:
                    return model.get_base_model().transformer.wte
                except Exception:
                    return model.transformer.wte
            else:
                try:
                    return model.get_base_model().model.embed_tokens
                except Exception:
                    return model.model.embed_tokens
        except AttributeError:
            if "pythia" in model_name:
                return model.gpt_neox.embed_in
            # Default to Llama-style
            return model.model.embed_tokens

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load trained CODI checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file or HuggingFace model ID
        """
        from safetensors.torch import load_file
        from huggingface_hub import hf_hub_download
        import os

        print(f"Loading CODI checkpoint from {checkpoint_path}...")

        # Check if it's a HuggingFace model ID (contains / and doesn't exist locally)
        if "/" in checkpoint_path and not os.path.exists(checkpoint_path):
            print(f"Downloading checkpoint from HuggingFace: {checkpoint_path}")
            try:
                # Try to download model.safetensors first
                checkpoint_file = hf_hub_download(
                    repo_id=checkpoint_path,
                    filename="model.safetensors"
                )
            except Exception:
                try:
                    # Fallback to pytorch_model.bin
                    checkpoint_file = hf_hub_download(
                        repo_id=checkpoint_path,
                        filename="pytorch_model.bin"
                    )
                except Exception as e:
                    raise ValueError(f"Could not download checkpoint from {checkpoint_path}: {e}")
        else:
            checkpoint_file = checkpoint_path

        # print(f"DEBUG: Loading from file: {checkpoint_file}")

        # Load the checkpoint
        if checkpoint_file.endswith('.safetensors'):
            state_dict = load_file(checkpoint_file)
        else:
            state_dict = torch.load(checkpoint_file, map_location=self.device_str)

        # print(f"DEBUG: Checkpoint contains {len(state_dict)} parameters")
        # print(f"DEBUG: Model expects {len(self.state_dict())} parameters")
        # print(f"DEBUG: Sample checkpoint keys: {list(state_dict.keys())[:5]}")
        # print(f"DEBUG: Sample model keys: {list(self.state_dict().keys())[:5]}")

        # Load with strict=False (matching original CODI) but validate the
        # result. All known CODI checkpoints (HF and local) store the full
        # state dict, so a correct load is 0 missing / 0 unexpected. Any
        # mismatch would silently leave a submodule -- e.g. the projection
        # MLP or LoRA adapters -- at random init and quietly produce wrong
        # results, so we fail loudly instead.
        result = self.load_state_dict(state_dict, strict=False)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(
                f"CODI checkpoint mismatch: {len(result.missing_keys)} missing, "
                f"{len(result.unexpected_keys)} unexpected. "
                f"missing={result.missing_keys[:10]} "
                f"unexpected={result.unexpected_keys[:10]}"
            )

        # Tie weights (like original CODI test.py line 81)
        self.codi.tie_weights()

        # Compute hash of trainable parameters (exactly matches original CODI test.py lines 343-352)
        import hashlib
        param_hash = hashlib.md5()
        trainable_params = []
        for name, param in sorted(self.named_parameters()):
            if param.requires_grad:
                trainable_params.append(name)
                param_bytes = param.detach().cpu().to(torch.float32).numpy().tobytes()
                param_hash.update(param_bytes)

        # print(f'DEBUG: Trainable parameters: {len(trainable_params)}')
        # print(f'DEBUG: First 10 trainable params:')
        # for name in trainable_params[:10]:
        #     param = dict(self.named_parameters())[name]
        #     print(f'  {name}: {param.shape} ({param.numel()} elements)')
        # print(f'DEBUG: Last 5 trainable params:')
        # for name in trainable_params[-5:]:
        #     param = dict(self.named_parameters())[name]
        #     print(f'  {name}: {param.shape} ({param.numel()} elements)')

        # # Count total trainable elements
        # total_elements = sum(dict(self.named_parameters())[name].numel() for name in trainable_params)
        # print(f'DEBUG: Total trainable elements: {total_elements:,}')
        # print(f'DEBUG: Vocab size: {self.codi.config.vocab_size}')
        # print(f'DEBUG: Embedding weight shape: {self.codi.get_base_model().transformer.wte.weight.shape}')
        # print(f'DEBUG: LM head weight shape: {self.codi.get_base_model().lm_head.weight.shape}')
        # print(f'DEBUG: Parameter hash: {param_hash.hexdigest()}')

        # print(f"Checkpoint loaded successfully")

    def encode_question(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> EncodedState:
        """
        Encode the question and return cached state for efficient ablation analysis.

        This allows running multiple ablation experiments without re-encoding
        the question each time.

        Args:
            input_ids: Question tokens [batch_size, seq_len] (including BOT token)
            attention_mask: Attention mask

        Returns:
            EncodedState with KV cache and initial latent embedding
        """
        with torch.no_grad():
            outputs = self.codi(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                output_hidden_states=True
            )

            past_key_values = outputs.past_key_values
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

            if self.use_prj:
                latent_embd = self.prj(latent_embd)

            # Deep copy KV cache to avoid mutation issues
            cached_kv = [(k.clone(), v.clone()) for k, v in past_key_values]

            return EncodedState(
                past_key_values=cached_kv,
                initial_latent_embd=latent_embd.clone()
            )

    def get_bot_logits_with_ablation(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        ablate_positions: Optional[List[int]] = None
    ) -> torch.Tensor:
        """
        Get logits at the BOT position with optional DIRECT ablation using attention masking.

        This masks attention ONLY from the BOT position to the ablated positions.
        Other positions can still attend to the ablated positions normally.

        This is used to analyze what question tokens contribute to the
        hidden state at the BOT position (where '24' appears for 4*6).

        Args:
            input_ids: Question tokens [batch_size, seq_len] (including BOT token)
            attention_mask: Attention mask
            ablate_positions: Positions to ablate (mask attention from BOT to these)

        Returns:
            Logits at the BOT position [batch_size, vocab_size]
        """
        if ablate_positions is None:
            ablate_positions = []

        with torch.no_grad():
            base_model = self.codi.get_base_model()

            if ablate_positions:
                # BOT is at the last position of the input
                bot_position = input_ids.shape[1] - 1

                # Patch attention to mask ONLY from BOT position to ablated positions
                original_attns = patch_attention_for_ablation(
                    self.codi, bot_position, ablate_positions
                )

                try:
                    # Forward pass with patched attention
                    # MUST use output_attentions=True to force eager attention (not SDPA)
                    outputs = self.codi(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                        output_attentions=True
                    )
                finally:
                    # Restore original attention
                    restore_attention(original_attns)

                # Get logits at BOT position
                bot_hidden = outputs.hidden_states[-1][:, -1, :]
                logits = base_model.lm_head(bot_hidden)
            else:
                # No ablation - standard forward pass
                outputs = self.codi(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
                bot_hidden = outputs.hidden_states[-1][:, -1, :]
                logits = base_model.lm_head(bot_hidden)

            return logits

    def get_bot_logits_with_total_ablation(
        self,
        input_ids: torch.Tensor,
        ablate_positions: Optional[List[int]] = None
    ) -> torch.Tensor:
        """
        Get logits at the BOT position with TOTAL ablation using attention masking.

        This method blocks ALL attention to the ablated positions throughout the
        entire forward pass. This captures indirect effects where information would
        otherwise flow through intermediate positions.

        Example:
            Direct only:   '4' --X--> BOT  (blocked)
                          '4' --> '.' --> BOT  (NOT blocked, info still flows)

            Total effect:  '4' --X--> BOT  (blocked)
                          '4' -X-> '.' --> BOT  (blocked, no info flows anywhere)

        Implementation: Patches the attention layers to zero out attention weights
        to the ablated positions for ALL query positions, then renormalizes.

        Args:
            input_ids: Question tokens [batch_size, seq_len] (including BOT token)
            ablate_positions: Positions to block attention TO

        Returns:
            Logits at the BOT position [batch_size, vocab_size]
        """
        if ablate_positions is None:
            ablate_positions = []

        with torch.no_grad():
            base_model = self.codi.get_base_model()

            if ablate_positions:
                # Patch attention to block attention to ablated positions from ALL query positions
                # We need to patch for each query position
                seq_len = input_ids.shape[1]

                # For total ablation, we block attention to ablated positions from ALL positions
                # We'll patch attention for all query positions at once by creating a custom forward
                original_attns = []

                for layer in base_model.transformer.h:
                    original_attn = layer.attn._attn
                    original_attns.append((layer.attn, original_attn))

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
                            # attn_output = modified_weights @ value
                            modified_output = torch.matmul(modified_weights, value)

                            return modified_output, modified_weights

                        return patched_attn

                    layer.attn._attn = make_patched_attn(original_attn, ablate_positions)

                try:
                    # Forward pass with patched attention
                    # MUST use output_attentions=True to force eager attention (not SDPA)
                    outputs = self.codi(
                        input_ids=input_ids,
                        output_hidden_states=True,
                        output_attentions=True  # Forces fallback from SDPA to eager attention
                    )
                finally:
                    # Restore original attention
                    restore_attention(original_attns)

            else:
                # No ablation - standard forward pass
                outputs = self.codi(
                    input_ids=input_ids,
                    output_hidden_states=True
                )

            # Get logits at BOT position (last position)
            bot_hidden = outputs.hidden_states[-1][:, -1, :]
            logits = base_model.lm_head(bot_hidden)

            return logits

    def run_latent_iterations(
        self,
        encoded_state: EncodedState,
        num_iterations: int,
        target_iteration: Optional[int] = None,
        source_positions: Optional[List[int]] = None
    ) -> LatentIterationResult:
        """
        Run latent iterations with optional DIRECT ablation using attention masking.

        This masks attention ONLY from the target iteration's query to the source positions.
        Other positions can still attend to the source positions normally.

        This is much faster than calling generate() for ablation analysis because:
        1. Question encoding is reused (not re-computed)
        2. Answer generation is skipped (not needed for analysis)

        Args:
            encoded_state: Pre-computed state from encode_question()
            num_iterations: Number of latent iterations to run
            target_iteration: Which iteration to apply ablation (None = no ablation)
            source_positions: Positions to ablate (mask attention from target to these)

        Returns:
            LatentIterationResult with logits from each iteration
        """
        if source_positions is None:
            source_positions = []

        # Clone the cached state to avoid mutation
        past_key_values = [(k.clone(), v.clone()) for k, v in encoded_state.past_key_values]
        latent_embd = encoded_state.initial_latent_embd.clone()

        iteration_logits = []

        with torch.no_grad():
            for i in range(num_iterations):
                # Determine if we apply ablation this iteration
                apply_ablation = (
                    target_iteration == i and
                    len(source_positions) > 0
                )

                if apply_ablation:
                    # Patch attention to mask from current query (position 0 in input chunk)
                    # to source positions in the full KV cache
                    original_attns = patch_attention_for_ablation(
                        self.codi, 0, source_positions  # Query is always at position 0 in single-token input
                    )

                    try:
                        outputs = self.codi(
                            inputs_embeds=latent_embd,
                            use_cache=True,
                            output_hidden_states=True,
                            output_attentions=True,  # Force eager attention
                            past_key_values=past_key_values
                        )
                    finally:
                        restore_attention(original_attns)
                else:
                    outputs = self.codi(
                        inputs_embeds=latent_embd,
                        use_cache=True,
                        output_hidden_states=True,
                        past_key_values=past_key_values
                    )

                # Collect logits
                iteration_logits.append(outputs.logits.clone())

                # Update state for next iteration
                past_key_values = outputs.past_key_values
                latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

                if self.use_prj:
                    latent_embd = self.prj(latent_embd)

        return LatentIterationResult(iteration_logits=iteration_logits)

    def run_latent_iterations_with_split_ablation(
        self,
        encoded_state: EncodedState,
        num_iterations: int,
        ablation_spec: Dict[int, List[Dict]]
    ) -> LatentIterationResult:
        """
        Run latent iterations with iteration-specific, layer-specific ablation.

        This is used for the "split" ablation experiment where latent tokens are
        defined as diagonal regions spanning different layer halves across iterations.

        The ablation_spec maps iteration index -> list of token groups to ablate.
        Each token group has {'positions': [...], 'layer_range': (start, end)}.

        Example for ablating latent_2 diagonal (Q = question_length):
            ablation_spec = {
                2: [{'positions': [Q+2], 'layer_range': (6, 12)}],  # Upper half at iter 2
                3: [{'positions': [Q+2], 'layer_range': (0, 6)}]    # Lower half at iter 3
            }

        Args:
            encoded_state: Pre-computed state from encode_question()
            num_iterations: Number of latent iterations to run
            ablation_spec: Dict mapping iteration_idx -> token_groups to ablate

        Returns:
            LatentIterationResult with logits from each iteration
        """
        # Clone the cached state to avoid mutation
        past_key_values = [(k.clone(), v.clone()) for k, v in encoded_state.past_key_values]
        latent_embd = encoded_state.initial_latent_embd.clone()

        iteration_logits = []

        with torch.no_grad():
            for i in range(num_iterations):
                # Check if this iteration has ablation
                token_groups = ablation_spec.get(i, [])

                if token_groups:
                    # Patch attention for this iteration's token groups (layer-specific)
                    original_attns = patch_attention_for_token_groups(
                        self.codi, token_groups, target_query_idx=0
                    )

                    try:
                        outputs = self.codi(
                            inputs_embeds=latent_embd,
                            use_cache=True,
                            output_hidden_states=True,
                            output_attentions=True,  # Force eager attention
                            past_key_values=past_key_values
                        )
                    finally:
                        restore_attention(original_attns)
                else:
                    outputs = self.codi(
                        inputs_embeds=latent_embd,
                        use_cache=True,
                        output_hidden_states=True,
                        past_key_values=past_key_values
                    )

                # Collect logits
                iteration_logits.append(outputs.logits.clone())

                # Update state for next iteration
                past_key_values = outputs.past_key_values
                latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

                if self.use_prj:
                    latent_embd = self.prj(latent_embd)

        return LatentIterationResult(iteration_logits=iteration_logits)

    def get_iteration_logits_with_total_ablation(
        self,
        input_ids: torch.Tensor,
        num_iterations: int,
        target_iteration: int,
        ablate_position: int,
        question_length: int
    ) -> torch.Tensor:
        """
        Get logits at a specific iteration with TOTAL ablation using attention masking.

        This method blocks ALL attention to the ablated position throughout the entire
        forward pass (encoding + all iterations). This captures indirect effects where
        information would otherwise flow through intermediate positions.

        Example:
            Direct only:   '4' --X--> latent_2  (blocked)
                          '4' --> latent_1 --> latent_2  (NOT blocked)

            Total effect:  '4' --X--> latent_2  (blocked)
                          '4' -X-> latent_1 --> latent_2  (blocked, no info flows anywhere)

        Args:
            input_ids: Question tokens [batch_size, seq_len] (including BOT token)
            num_iterations: Total number of latent iterations
            target_iteration: Which iteration to get logits from (0-indexed)
            ablate_position: Position to ablate (in the full sequence including latents)
            question_length: Length of question (including BOT token)

        Returns:
            Logits at the target iteration [batch_size, 1, vocab_size]
        """
        with torch.no_grad():
            base_model = self.codi.get_base_model()

            # Create attention masking function that blocks attention to ablate_position
            def make_patched_attn(orig_attn, src_pos):
                def patched_attn(query, key, value, attention_mask=None, head_mask=None):
                    attn_output, attn_weights = orig_attn(query, key, value, attention_mask, head_mask)

                    # attn_weights shape: [batch, heads, query_len, key_len]
                    if src_pos < attn_weights.shape[3]:
                        modified_weights = attn_weights.clone()
                        modified_weights[:, :, :, src_pos] = 0

                        # Renormalize
                        row_sum = modified_weights.sum(dim=-1, keepdim=True)
                        row_sum = torch.clamp(row_sum, min=1e-9)
                        modified_weights = modified_weights / row_sum

                        # Recompute output
                        modified_output = torch.matmul(modified_weights, value)
                        return modified_output, modified_weights

                    return attn_output, attn_weights
                return patched_attn

            # Patch attention layers
            original_attns = []
            for layer in base_model.transformer.h:
                original_attn = layer.attn._attn
                original_attns.append((layer.attn, original_attn))
                layer.attn._attn = make_patched_attn(original_attn, ablate_position)

            try:
                # Encode question with attention masking
                # MUST use output_attentions=True to force eager attention
                outputs = self.codi(
                    input_ids=input_ids,
                    use_cache=True,
                    output_hidden_states=True,
                    output_attentions=True
                )

                past_key_values = outputs.past_key_values
                latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

                if self.use_prj:
                    latent_embd = self.prj(latent_embd)

                # Run iterations with attention masking still active
                for i in range(num_iterations):
                    outputs = self.codi(
                        inputs_embeds=latent_embd,
                        use_cache=True,
                        output_hidden_states=True,
                        output_attentions=True,  # Keep forcing eager attention
                        past_key_values=past_key_values
                    )

                    if i == target_iteration:
                        return outputs.logits

                    past_key_values = outputs.past_key_values
                    latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

                    if self.use_prj:
                        latent_embd = self.prj(latent_embd)

            finally:
                # Restore original attention
                restore_attention(original_attns)

        raise ValueError(f"target_iteration {target_iteration} not reached in {num_iterations} iterations")

    def get_iteration_logits_baseline(
        self,
        input_ids: torch.Tensor,
        num_iterations: int,
        target_iteration: int
    ) -> torch.Tensor:
        """
        Get baseline logits at a specific iteration (no ablation).

        This is used as the reference for computing KL divergence in total ablation analysis.

        Args:
            input_ids: Question tokens [batch_size, seq_len] (including BOT token)
            num_iterations: Total number of latent iterations
            target_iteration: Which iteration to get logits from (0-indexed)

        Returns:
            Logits at the target iteration [batch_size, 1, vocab_size]
        """
        with torch.no_grad():
            # Encode question
            outputs = self.codi(
                input_ids=input_ids,
                use_cache=True,
                output_hidden_states=True
            )

            past_key_values = outputs.past_key_values
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

            if self.use_prj:
                latent_embd = self.prj(latent_embd)

            # Run iterations
            for i in range(num_iterations):
                outputs = self.codi(
                    inputs_embeds=latent_embd,
                    use_cache=True,
                    output_hidden_states=True,
                    past_key_values=past_key_values
                )

                if i == target_iteration:
                    return outputs.logits

                past_key_values = outputs.past_key_values
                latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

                if self.use_prj:
                    latent_embd = self.prj(latent_embd)

        raise ValueError(f"target_iteration {target_iteration} not reached in {num_iterations} iterations")

    def get_answer_logits_with_ablation(
        self,
        input_ids: torch.Tensor,
        num_iterations: int,
        target_answer_position: int,
        ablate_positions: List[int],
        num_answer_tokens: int = 5
    ) -> torch.Tensor:
        """
        Get logits at a specific answer position with direct ablation (zeroing KV cache values).

        The position layout is:
        - 0 to question_length-1: question tokens (including BOT)
        - question_length to question_length+num_iterations-1: latent iterations
        - question_length+num_iterations: EOT
        - question_length+num_iterations+1: "The"
        - question_length+num_iterations+2: " answer"
        - question_length+num_iterations+3: " is"
        - question_length+num_iterations+4: ":"
        - question_length+num_iterations+5+: answer tokens

        Args:
            input_ids: Question tokens [1, question_length] (including BOT token)
            num_iterations: Number of latent iterations
            target_answer_position: Absolute position to get logits for
            ablate_positions: Positions to ablate (zero out KV cache values)
            num_answer_tokens: Max answer tokens to generate before target

        Returns:
            Logits at the target position [1, vocab_size]
        """
        device = input_ids.device
        question_length = input_ids.shape[1]

        with torch.no_grad():
            # 1. Encode question
            outputs = self.codi(
                input_ids=input_ids,
                use_cache=True,
                output_hidden_states=True
            )

            past_key_values = outputs.past_key_values
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

            if self.use_prj:
                latent_embd = self.prj(latent_embd)

            # 2. Run latent iterations
            for i in range(num_iterations):
                outputs = self.codi(
                    inputs_embeds=latent_embd,
                    use_cache=True,
                    output_hidden_states=True,
                    past_key_values=past_key_values
                )

                past_key_values = outputs.past_key_values
                latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

                if self.use_prj:
                    latent_embd = self.prj(latent_embd)

            # 3. Add EOT
            embedding_layer = self.get_embd(self.codi, self.model_name)
            eot_emb = embedding_layer(
                torch.tensor([[self.eot_id]], device=device)
            )

            outputs = self.codi(
                inputs_embeds=eot_emb,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values
            )
            past_key_values = outputs.past_key_values
            current_pos = question_length + num_iterations + 1  # After EOT

            # 4. Teacher-force "The answer is:"
            teacher_tokens = self.tokenizer.encode("The answer is:", add_special_tokens=False)
            for token_id in teacher_tokens:
                token_emb = embedding_layer(
                    torch.tensor([[token_id]], device=device)
                )
                outputs = self.codi(
                    inputs_embeds=token_emb,
                    use_cache=True,
                    output_hidden_states=True,
                    past_key_values=past_key_values
                )
                past_key_values = outputs.past_key_values

                # Check if this is the target position
                if current_pos == target_answer_position:
                    # Apply ablation for this position
                    if ablate_positions:
                        ablated_kv = ablate_kv_cache_positions(past_key_values, ablate_positions)
                    else:
                        ablated_kv = past_key_values

                    # Re-run to get logits with ablation applied
                    ablated_outputs = self.codi(
                        inputs_embeds=token_emb,
                        use_cache=False,
                        output_hidden_states=False,
                        past_key_values=ablated_kv
                    )
                    return ablated_outputs.logits[0, -1, :]

                current_pos += 1

            # 5. Generate answer tokens until we reach target position
            last_hidden = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

            for ans_idx in range(num_answer_tokens):
                # Get next token logits
                outputs = self.codi(
                    inputs_embeds=last_hidden,
                    use_cache=True,
                    output_hidden_states=True,
                    past_key_values=past_key_values
                )

                # Check if this is the target position
                if current_pos == target_answer_position:
                    # Apply ablation
                    if ablate_positions:
                        ablated_kv = ablate_kv_cache_positions(past_key_values, ablate_positions)
                    else:
                        ablated_kv = past_key_values

                    # Re-run to get logits with ablation applied
                    ablated_outputs = self.codi(
                        inputs_embeds=last_hidden,
                        use_cache=False,
                        output_hidden_states=False,
                        past_key_values=ablated_kv
                    )
                    return ablated_outputs.logits[0, -1, :]

                # Update for next iteration
                past_key_values = outputs.past_key_values
                next_token_logits = outputs.logits[:, -1, :self.codi.config.vocab_size-1]
                next_token_id = torch.argmax(next_token_logits, dim=-1)
                last_hidden = embedding_layer(next_token_id).unsqueeze(1)
                current_pos += 1

            raise ValueError(f"target_answer_position {target_answer_position} not reached")

    def get_answer_logits_with_total_ablation(
        self,
        input_ids: torch.Tensor,
        num_iterations: int,
        target_answer_position: int,
        ablate_positions: List[int],
        num_answer_tokens: int = 5
    ) -> torch.Tensor:
        """
        Get logits at a specific answer position with total ablation (attention masking).

        This blocks ALL positions from attending to the ablated positions, not just
        the target position. This captures indirect/multi-hop attention effects.

        Args:
            input_ids: Question tokens [1, question_length] (including BOT token)
            num_iterations: Number of latent iterations
            target_answer_position: Absolute position to get logits for
            ablate_positions: Positions to ablate (block attention to)
            num_answer_tokens: Max answer tokens to generate before target

        Returns:
            Logits at the target position [1, vocab_size]
        """
        device = input_ids.device
        question_length = input_ids.shape[1]

        with torch.no_grad():
            # 1. Encode question with ablation
            if ablate_positions:
                original_attns = patch_attention_total_ablation(
                    self.codi, ablate_positions
                )

            try:
                outputs = self.codi(
                    input_ids=input_ids,
                    use_cache=True,
                    output_hidden_states=True,
                    output_attentions=True if ablate_positions else False
                )
            finally:
                if ablate_positions:
                    restore_attention(original_attns)

            past_key_values = outputs.past_key_values
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

            if self.use_prj:
                latent_embd = self.prj(latent_embd)

            # 2. Run latent iterations with ablation
            for i in range(num_iterations):
                if ablate_positions:
                    original_attns = patch_attention_total_ablation(
                        self.codi, ablate_positions
                    )

                try:
                    outputs = self.codi(
                        inputs_embeds=latent_embd,
                        use_cache=True,
                        output_hidden_states=True,
                        output_attentions=True if ablate_positions else False,
                        past_key_values=past_key_values
                    )
                finally:
                    if ablate_positions:
                        restore_attention(original_attns)

                past_key_values = outputs.past_key_values
                latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

                if self.use_prj:
                    latent_embd = self.prj(latent_embd)

            # 3. Add EOT with ablation
            embedding_layer = self.get_embd(self.codi, self.model_name)
            eot_emb = embedding_layer(
                torch.tensor([[self.eot_id]], device=device)
            )

            if ablate_positions:
                original_attns = patch_attention_total_ablation(
                    self.codi, ablate_positions
                )

            try:
                outputs = self.codi(
                    inputs_embeds=eot_emb,
                    use_cache=True,
                    output_hidden_states=True,
                    output_attentions=True if ablate_positions else False,
                    past_key_values=past_key_values
                )
            finally:
                if ablate_positions:
                    restore_attention(original_attns)

            past_key_values = outputs.past_key_values
            current_pos = question_length + num_iterations + 1  # After EOT

            # 4. Teacher-force "The answer is:" with ablation
            teacher_tokens = self.tokenizer.encode("The answer is:", add_special_tokens=False)
            for token_id in teacher_tokens:
                token_emb = embedding_layer(
                    torch.tensor([[token_id]], device=device)
                )

                if ablate_positions:
                    original_attns = patch_attention_total_ablation(
                        self.codi, ablate_positions
                    )

                try:
                    outputs = self.codi(
                        inputs_embeds=token_emb,
                        use_cache=True,
                        output_hidden_states=True,
                        output_attentions=True if ablate_positions else False,
                        past_key_values=past_key_values
                    )
                finally:
                    if ablate_positions:
                        restore_attention(original_attns)

                past_key_values = outputs.past_key_values

                if current_pos == target_answer_position:
                    return outputs.logits[0, -1, :]

                current_pos += 1

            # 5. Generate answer tokens until we reach target position
            last_hidden = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

            for ans_idx in range(num_answer_tokens):
                if ablate_positions:
                    original_attns = patch_attention_total_ablation(
                        self.codi, ablate_positions
                    )

                try:
                    outputs = self.codi(
                        inputs_embeds=last_hidden,
                        use_cache=True,
                        output_hidden_states=True,
                        output_attentions=True if ablate_positions else False,
                        past_key_values=past_key_values
                    )
                finally:
                    if ablate_positions:
                        restore_attention(original_attns)

                if current_pos == target_answer_position:
                    return outputs.logits[0, -1, :]

                past_key_values = outputs.past_key_values
                next_token_logits = outputs.logits[:, -1, :self.codi.config.vocab_size-1]
                next_token_id = torch.argmax(next_token_logits, dim=-1)
                last_hidden = embedding_layer(next_token_id).unsqueeze(1)
                current_pos += 1

            raise ValueError(f"target_answer_position {target_answer_position} not reached")

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 64,
        num_iterations: Optional[int] = None,
        greedy: bool = True,  # Match original CODI
        temperature: float = 0.1,  # Match original CODI
        top_k: int = 40,  # Match original CODI
        top_p: float = 0.95,  # Match original CODI
        teacher_force_prefix: Optional[str] = None,  # Teacher-force prefix after EOT
        ablation_config: Optional[Dict] = None  # Ablation configuration
    ):
        """
        Generate answer using CODI's iterative latent reasoning.

        Args:
            input_ids: Question tokens [batch_size, seq_len]
            attention_mask: Attention mask
            max_new_tokens: Maximum answer tokens to generate
            num_iterations: Number of latent iterations (defaults to self.num_latent)
            greedy: Whether to use greedy decoding (True) or sampling (False)
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p (nucleus) sampling
            teacher_force_prefix: Optional prefix to teacher-force after EOT (e.g., "The answer is:")
            ablation_config: Optional dict with:
                - 'target_iteration': int - which iteration to ablate (0-indexed)
                - 'source_positions': List[int] - which positions to mask attention to
                - 'target_layer': int - specific layer to ablate (None = all layers)
                - 'collect_iteration_logits': bool - whether to collect logits at each iteration

        Returns:
            If ablation_config is provided: GenerateResult with output_ids and optional iteration_logits
            Otherwise: torch.Tensor of generated token IDs [batch_size, seq_len + new_tokens]
        """
        if num_iterations is None:
            num_iterations = self.num_latent

        batch_size = input_ids.shape[0]
        device = input_ids.device

        # Parse ablation config
        target_iteration = ablation_config.get('target_iteration') if ablation_config else None
        source_positions = ablation_config.get('source_positions', []) if ablation_config else []
        target_layer = ablation_config.get('target_layer') if ablation_config else None
        collect_iteration_logits = ablation_config.get('collect_iteration_logits', False) if ablation_config else False

        iteration_logits = [] if collect_iteration_logits else None

        with torch.no_grad():
            # 1. Encode question
            outputs = self.codi(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                output_hidden_states=True
            )

            past_key_values = outputs.past_key_values
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

            if self.use_prj:
                latent_embd = self.prj(latent_embd)

            # Track the current KV cache length (positions we can attend to)
            # After encoding question: kv_cache_len = input_ids.shape[1]
            kv_cache_len = input_ids.shape[1]

            # 2. Iteratively generate latent embeddings
            for i in range(num_iterations):
                # Determine if we apply ablation this iteration
                apply_ablation = (
                    ablation_config is not None and
                    target_iteration == i and
                    len(source_positions) > 0
                )

                if apply_ablation:
                    # Patch attention to mask from current query to source positions
                    original_attns = patch_attention_for_ablation(
                        self.codi, 0, source_positions, target_layer  # Query is at position 0 in single-token input
                    )

                    try:
                        outputs = self.codi(
                            inputs_embeds=latent_embd,
                            use_cache=True,
                            output_hidden_states=True,
                            output_attentions=True,  # Force eager attention
                            past_key_values=past_key_values
                        )
                    finally:
                        restore_attention(original_attns)
                else:
                    outputs = self.codi(
                        inputs_embeds=latent_embd,
                        use_cache=True,
                        output_hidden_states=True,
                        past_key_values=past_key_values
                    )

                # Collect iteration logits if requested
                if iteration_logits is not None:
                    iteration_logits.append(outputs.logits.clone())

                # IMPORTANT: Always update from the model's output KV cache
                # (ablation only affects this iteration's computation)
                past_key_values = outputs.past_key_values
                kv_cache_len += 1  # We added one position

                latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

                if self.use_prj:
                    latent_embd = self.prj(latent_embd)

            # 2.5. Create EOT (End of Thought) embedding to signal transition to answer
            embedding_layer = self.get_embd(self.codi, self.model_name)
            eot_emb = embedding_layer(
                torch.tensor([self.eot_id], dtype=torch.long, device=device)
            ).unsqueeze(0)
            eot_emb = eot_emb.expand(batch_size, -1, -1)
            output = eot_emb

            # 2.6. If teacher-forcing prefix provided, add it to KV cache
            teacher_forced_ids = []
            if teacher_force_prefix is not None:
                # Tokenize the prefix
                prefix_ids = self.tokenizer.encode(teacher_force_prefix, add_special_tokens=False)
                teacher_forced_ids = prefix_ids.copy()

                # Process each token through the model to extend KV cache
                for prefix_token_id in prefix_ids:
                    # Forward pass with current output embedding
                    outputs = self.codi(
                        inputs_embeds=output,
                        use_cache=True,
                        past_key_values=past_key_values,
                        attention_mask=None
                    )

                    # Update KV cache
                    past_key_values = outputs.past_key_values

                    # Get embedding for the teacher-forced token
                    output = embedding_layer(
                        torch.tensor([prefix_token_id], dtype=torch.long, device=device)
                    ).unsqueeze(0).expand(batch_size, -1, -1)

            # 3. Generate answer tokens
            generated_ids = []
            gen_count = 0
            for _ in range(max_new_tokens):
                gen_count += 1
                outputs = self.codi(
                    inputs_embeds=output,
                    use_cache=True,
                    past_key_values=past_key_values,
                    attention_mask=None
                )

                # Get logits for next token
                # Note: We slice to exclude only the eot token from generation
                # self.codi.config.vocab_size includes the 3 special tokens we added
                # Slice to vocab_size-1 to include original vocab + pad + bot, but exclude eot
                # This matches the original CODI implementation
                next_token_logits = outputs.logits[:, -1, :self.codi.config.vocab_size-1]

                # Sampling logic - EXACT COPY from original CODI test.py
                if greedy:
                    next_token_id = torch.argmax(next_token_logits, dim=-1)  # [batch_size]
                else:
                    next_token_logits /= temperature
                    if top_k > 1:
                        top_k_values, _ = torch.topk(next_token_logits, top_k, dim=-1)
                        min_top_k_value = top_k_values[:, -1].unsqueeze(-1)
                        next_token_logits[next_token_logits < min_top_k_value] = float('-inf')

                    if top_p < 1.0:
                        sorted_logit, sorted_indices = torch.sort(next_token_logits, descending=True, dim=-1)
                        cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logit, dim=-1), dim=-1)

                        sorted_indices_to_remove = cumulative_probs > top_p
                        if sorted_indices_to_remove.any():
                            sorted_indices_to_remove = sorted_indices_to_remove.roll(1, dims=-1)
                            sorted_indices_to_remove[:, 0] = False

                        for b in range(next_token_logits.size(0)):
                            next_token_logits[b, sorted_indices[b, sorted_indices_to_remove[b]]] = float('-inf')

                    probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                    next_token_id = torch.multinomial(probs, num_samples=1).squeeze(-1)

                # Check for EOS or EOT (handle batch dimension)
                # For batch_size=1, check if generated token is EOS or Llama's EOT
                if batch_size == 1:
                    token_id = next_token_id[0].item()
                    if token_id == self.tokenizer.eos_token_id:
                        break
                    if self.llama_eot_token_id is not None and token_id == self.llama_eot_token_id:
                        break

                generated_ids.append(next_token_id.unsqueeze(1))  # [batch_size, 1]

                # Update embeddings for next iteration
                # next_token_id is [batch_size], embedding returns [batch_size, hidden_dim]
                # unsqueeze(1) makes it [batch_size, 1, hidden_dim]
                output = embedding_layer(next_token_id).unsqueeze(1)

                past_key_values = outputs.past_key_values

        # Construct output with placeholder tokens for visualization
        # The actual forward pass used embeddings, but we insert token IDs here
        # so humans can see where latent iterations occurred

        # Create placeholder tokens for latent iterations
        latent_placeholder = torch.full(
            (batch_size, num_iterations),
            self.latent_id,  # Use <|latent|> token for visualization (tokenizer-only, not in model vocab)
            dtype=torch.long,
            device=device
        )

        # Create EOT token
        eot_token = torch.full(
            (batch_size, 1),
            self.eot_id,
            dtype=torch.long,
            device=device
        )

        # Concatenate: question + bot + latent_placeholders + eot + teacher_forced + answer
        output_parts = [
            input_ids,              # Question + bot
            latent_placeholder,     # Latent iterations (placeholders)
            eot_token,              # EOT marker
        ]

        # Add teacher-forced tokens if present
        if teacher_forced_ids:
            teacher_forced_tensor = torch.tensor(
                teacher_forced_ids, dtype=torch.long, device=device
            ).unsqueeze(0).expand(batch_size, -1)
            output_parts.append(teacher_forced_tensor)

        # Add generated answer tokens
        if generated_ids:
            generated_tensor = torch.cat(generated_ids, dim=1)
            output_parts.append(generated_tensor)

        full_output = torch.cat(output_parts, dim=1)

        # Return result based on whether ablation was used
        if ablation_config is not None:
            return GenerateResult(
                output_ids=full_output,
                iteration_logits=iteration_logits
            )
        else:
            return full_output


def create_inference_model(
    model_path: str = "gpt2",
    num_latent: int = 6,
    use_prj: bool = True,
    prj_dim: int = None,
    use_lora: bool = True,
    lora_r: int = 128,
    lora_alpha: int = 32,
    device: str = "cuda"
) -> CODIInference:
    """
    Factory function to create CODI inference model.

    Args:
        model_path: Base model path (default: gpt2)
        num_latent: Number of latent iterations
        use_prj: Whether to use projection layer
        prj_dim: Projection dimension (None = use model's hidden size)
        use_lora: Whether to use LoRA
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        device: Device to load on

    Returns:
        CODI inference model
    """
    model = CODIInference(
        model_name_or_path=model_path,
        num_latent=num_latent,
        use_prj=use_prj,
        prj_dim=prj_dim,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        device=device
    )
    model = model.to(device)
    model.eval()
    return model
