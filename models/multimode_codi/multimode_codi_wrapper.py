"""
Multi-mode CODI inference wrapper.

Supports three inference modes with different token configurations:
- Direct: prompt + eot_id + eocot_id → answer (no reasoning)
- Verbalized: prompt + eot_id + bocot_id → [CoT tokens] + eocot_id + answer
- Latent: prompt + eot_id + bocot_id → [N latent iterations] → eocot_id → answer

Token Assignment:
- GPT-2: Only 2 new tokens (bocot_id, eocot_id = vocab_size + 0,1), pad_token reuses eos_token
- Llama-3.x: Has built-in <|eot_id|>, only 2 new tokens (bocot, eocot), pad reuses eos
"""

import torch
import torch.nn as nn
from typing import Optional, NamedTuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType


class GenerateResult(NamedTuple):
    """Result from multimode CODI generate."""
    output_ids: torch.Tensor  # Generated token IDs
    mode: str  # Which mode was used


class MultimodeCODIInference(nn.Module):
    """
    Multi-mode CODI model for inference.

    Supports three modes:
    - direct: No reasoning, direct answer generation
    - verbalized: Standard autoregressive CoT generation
    - latent: Hidden state replacement (iterative latent reasoning)
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
        Initialize multi-mode CODI for inference.

        Args:
            model_name_or_path: Base model path (e.g., "gpt2" or "meta-llama/Llama-3.2-1B-Instruct")
            num_latent: Number of latent iterations for latent mode
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
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            resume_download=True
        )

        # Get original vocab size before any modifications
        ori_vocab_size = self.model.config.vocab_size
        self.ori_vocab_size = ori_vocab_size

        # Assign special tokens based on model type
        self._setup_special_tokens(model_name_or_path, ori_vocab_size)

        # Resize embeddings for new tokens
        new_vocab_size = ori_vocab_size + self.num_new_tokens
        self.model.resize_token_embeddings(new_vocab_size)

        # Model dimensions
        self.dim = self.model.config.hidden_size

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
            self.model = get_peft_model(self.model, lora_config)
            print("LoRA applied successfully")

        # Load tokenizer
        model_type = self.model.config.model_type
        if model_type == "gpt2":
            tokenizer_name = "gpt2"
        else:
            tokenizer_name = self.model_name

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)

        # Set padding side to LEFT for consistent position handling
        self.tokenizer.padding_side = "left"

        # Setup pad token if needed - reuse eos_token (matches training code)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pad_token_id = self.tokenizer.pad_token_id

        # Projection layer (optional)
        if use_prj:
            self.prj = nn.Sequential(
                nn.Dropout(prj_dropout),
                nn.Linear(self.dim, prj_dim),
                nn.GELU(),
                nn.Linear(prj_dim, self.dim)
            )
            if not prj_no_ln:
                self.prj.add_module("ln", nn.LayerNorm(self.dim))
            self.prj = self.prj.to(torch.bfloat16)

        print("Multi-mode CODI inference model initialized")

    def _setup_special_tokens(self, model_name: str, ori_vocab_size: int):
        """
        Setup special token IDs based on model type.

        For GPT-2: No eot token, only 2 new tokens (bocot, eocot), pad reuses eos
        For Llama-3.x: <|eot_id|> is built-in, only 2 new tokens (bocot, eocot), pad reuses eos
        """
        model_name_lower = model_name.lower()

        # Determine if we should use eot token based on model type
        self.use_eot = "llama" in model_name_lower

        if self.use_eot:
            # Llama-3.x has built-in <|eot_id|>
            # Try to get it from the tokenizer
            temp_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
            try:
                self.eot_id = temp_tokenizer.convert_tokens_to_ids("<|eot_id|>")
                if self.eot_id == temp_tokenizer.unk_token_id:
                    # Fallback if not found - add it as new token
                    self.eot_id = ori_vocab_size
                    self.bocot_id = ori_vocab_size + 1
                    self.eocot_id = ori_vocab_size + 2
                    self.pad_token_id = None  # Will be set to eos_token_id later
                    self.num_new_tokens = 3
                else:
                    # Use built-in eot_id, add only 2 new tokens (bocot, eocot)
                    # pad_token will reuse eos_token (matches training code)
                    self.bocot_id = ori_vocab_size
                    self.eocot_id = ori_vocab_size + 1
                    self.pad_token_id = None  # Will be set to eos_token_id later
                    self.num_new_tokens = 2
            except Exception:
                # Fallback: treat all as new
                self.eot_id = ori_vocab_size
                self.bocot_id = ori_vocab_size + 1
                self.eocot_id = ori_vocab_size + 2
                self.pad_token_id = None  # Will be set to eos_token_id later
                self.num_new_tokens = 3
        else:
            # GPT-2 and other models: no eot token, only 2 new tokens (bocot, eocot)
            # pad_token will reuse eos_token (matches training code)
            self.eot_id = None  # No eot token for GPT-2
            self.bocot_id = ori_vocab_size
            self.eocot_id = ori_vocab_size + 1
            self.pad_token_id = None  # Will be set to eos_token_id later
            self.num_new_tokens = 2

        # Latent token ID (used for visualization, same as bocot_id)
        self.latent_id = self.bocot_id

        print(f"Special tokens - use_eot: {self.use_eot}, eot_id: {self.eot_id}, bocot_id: {self.bocot_id}, "
              f"eocot_id: {self.eocot_id}, pad_token_id: {self.pad_token_id}")

    def get_embd(self, model, model_name):
        """Get embedding layer from model."""
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
            return model.model.embed_tokens

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load trained checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file or HuggingFace model ID
        """
        from safetensors.torch import load_file
        from huggingface_hub import hf_hub_download
        import os

        print(f"Loading checkpoint from {checkpoint_path}...")

        # Check if it's a HuggingFace model ID
        if "/" in checkpoint_path and not os.path.exists(checkpoint_path):
            print(f"Downloading checkpoint from HuggingFace: {checkpoint_path}")
            try:
                checkpoint_file = hf_hub_download(
                    repo_id=checkpoint_path,
                    filename="model.safetensors"
                )
            except Exception:
                try:
                    checkpoint_file = hf_hub_download(
                        repo_id=checkpoint_path,
                        filename="pytorch_model.bin"
                    )
                except Exception as e:
                    raise ValueError(f"Could not download checkpoint from {checkpoint_path}: {e}")
        else:
            checkpoint_file = checkpoint_path

        # Load the checkpoint
        if checkpoint_file.endswith('.safetensors'):
            state_dict = load_file(checkpoint_file)
        else:
            state_dict = torch.load(checkpoint_file, map_location=self.device_str)

        # Remap keys from CODI format (codi.*) to coconut_interp format (model.*)
        # CODI repo stores model as self.codi, but this wrapper uses self.model
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace("codi.", "model.")
            new_state_dict[new_key] = value
        state_dict = new_state_dict

        # Load with strict=False to get detailed error info, then validate
        result = self.load_state_dict(state_dict, strict=False)

        # Validate checkpoint loaded correctly - don't allow silent fallback to base LLM
        if result.unexpected_keys:
            raise RuntimeError(
                f"Checkpoint has {len(result.unexpected_keys)} unexpected keys. "
                f"This likely indicates a key format mismatch. "
                f"First 5 unexpected: {result.unexpected_keys[:5]}"
            )
        if result.missing_keys:
            raise RuntimeError(
                f"Checkpoint is missing {len(result.missing_keys)} keys. "
                f"This likely indicates an incompatible checkpoint. "
                f"First 5 missing: {result.missing_keys[:5]}"
            )

        # Tie weights
        self.model.tie_weights()

        print("Checkpoint loaded successfully")

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        mode: str = "latent",
        max_new_tokens: int = 64,
        num_iterations: Optional[int] = None,
        greedy: bool = True,
        temperature: float = 0.1,
        top_k: int = 40,
        top_p: float = 0.95,
    ) -> GenerateResult:
        """
        Generate in the specified mode.

        Args:
            input_ids: Input token IDs (should NOT include eot_id, bocot_id, etc. - we add them)
            attention_mask: Attention mask
            mode: One of "direct", "verbalized", "latent"
            max_new_tokens: Maximum answer tokens to generate
            num_iterations: Number of latent iterations (for latent mode)
            greedy: Whether to use greedy decoding
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p sampling

        Returns:
            GenerateResult with output_ids and mode
        """
        if num_iterations is None:
            num_iterations = self.num_latent

        if mode == "direct":
            return self._generate_direct(
                input_ids, attention_mask, max_new_tokens, greedy, temperature, top_k, top_p
            )
        elif mode == "verbalized":
            return self._generate_verbalized(
                input_ids, attention_mask, max_new_tokens, greedy, temperature, top_k, top_p
            )
        elif mode == "latent":
            return self._generate_latent(
                input_ids, attention_mask, max_new_tokens, num_iterations, greedy, temperature, top_k, top_p
            )
        else:
            raise ValueError(f"Unknown mode: {mode}. Must be 'direct', 'verbalized', or 'latent'")

    def _generate_direct(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
        greedy: bool,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> GenerateResult:
        """
        Direct mode: prompt + [eot_id] + eocot_id → answer (no reasoning)
        eot_id is only included for Llama (use_eot=True).
        """
        device = input_ids.device
        batch_size = input_ids.shape[0]

        with torch.no_grad():
            embedding_layer = self.get_embd(self.model, self.model_name)

            # Get embeddings for input
            inputs_embeds = embedding_layer(input_ids)

            # Build embedding sequence: input + [eot] + eocot
            embed_parts = [inputs_embeds]
            output_parts = [input_ids]

            if self.use_eot:
                eot_emb = embedding_layer(
                    torch.tensor([[self.eot_id]], device=device).expand(batch_size, -1)
                )
                embed_parts.append(eot_emb)
                eot_tensor = torch.full((batch_size, 1), self.eot_id, dtype=torch.long, device=device)
                output_parts.append(eot_tensor)

            eocot_emb = embedding_layer(
                torch.tensor([[self.eocot_id]], device=device).expand(batch_size, -1)
            )
            embed_parts.append(eocot_emb)
            eocot_tensor = torch.full((batch_size, 1), self.eocot_id, dtype=torch.long, device=device)
            output_parts.append(eocot_tensor)

            # Concatenate embeddings
            current_embeds = torch.cat(embed_parts, dim=1)

            # Generate answer tokens
            generated_ids = self._autoregressive_generate(
                current_embeds, max_new_tokens, greedy, temperature, top_k, top_p
            )

            # Construct output
            output_parts.append(generated_ids)
            output_ids = torch.cat(output_parts, dim=1)

            return GenerateResult(output_ids=output_ids, mode="direct")

    def _generate_verbalized(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
        greedy: bool,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> GenerateResult:
        """
        Verbalized mode: prompt + [eot_id] + bocot_id → [CoT tokens] + eocot_id + answer
        Standard autoregressive generation of explicit CoT.
        eot_id is only included for Llama (use_eot=True).
        """
        device = input_ids.device
        batch_size = input_ids.shape[0]

        with torch.no_grad():
            embedding_layer = self.get_embd(self.model, self.model_name)

            # Get embeddings for input
            inputs_embeds = embedding_layer(input_ids)

            # Build embedding sequence: input + [eot] + bocot
            embed_parts = [inputs_embeds]
            output_parts = [input_ids]

            if self.use_eot:
                eot_emb = embedding_layer(
                    torch.tensor([[self.eot_id]], device=device).expand(batch_size, -1)
                )
                embed_parts.append(eot_emb)
                eot_tensor = torch.full((batch_size, 1), self.eot_id, dtype=torch.long, device=device)
                output_parts.append(eot_tensor)

            bocot_emb = embedding_layer(
                torch.tensor([[self.bocot_id]], device=device).expand(batch_size, -1)
            )
            embed_parts.append(bocot_emb)
            bocot_tensor = torch.full((batch_size, 1), self.bocot_id, dtype=torch.long, device=device)
            output_parts.append(bocot_tensor)

            # Concatenate embeddings
            current_embeds = torch.cat(embed_parts, dim=1)

            # Generate CoT + eocot + answer autoregressively
            generated_ids = self._autoregressive_generate(
                current_embeds, max_new_tokens, greedy, temperature, top_k, top_p
            )

            # Construct output
            output_parts.append(generated_ids)
            output_ids = torch.cat(output_parts, dim=1)

            return GenerateResult(output_ids=output_ids, mode="verbalized")

    def _generate_latent(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
        num_iterations: int,
        greedy: bool,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> GenerateResult:
        """
        Latent mode: prompt + [eot_id] + bocot_id → [N latent iterations] → eocot_id → answer
        Uses hidden state replacement (iterative latent reasoning).
        eot_id is only included for Llama (use_eot=True).
        """
        device = input_ids.device
        batch_size = input_ids.shape[0]

        with torch.no_grad():
            embedding_layer = self.get_embd(self.model, self.model_name)

            # Build embedding sequence: input + [eot] + bocot
            inputs_embeds = embedding_layer(input_ids)
            embed_parts = [inputs_embeds]

            if self.use_eot:
                eot_emb = embedding_layer(
                    torch.tensor([[self.eot_id]], device=device).expand(batch_size, -1)
                )
                embed_parts.append(eot_emb)

            bocot_emb = embedding_layer(
                torch.tensor([[self.bocot_id]], device=device).expand(batch_size, -1)
            )
            embed_parts.append(bocot_emb)

            # Initial forward pass with input + [eot] + bocot
            current_embeds = torch.cat(embed_parts, dim=1)

            outputs = self.model(
                inputs_embeds=current_embeds,
                use_cache=True,
                output_hidden_states=True
            )

            past_key_values = outputs.past_key_values
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

            if self.use_prj:
                latent_embd = self.prj(latent_embd)

            # Run latent iterations
            for i in range(num_iterations):
                outputs = self.model(
                    inputs_embeds=latent_embd,
                    use_cache=True,
                    output_hidden_states=True,
                    past_key_values=past_key_values
                )

                past_key_values = outputs.past_key_values
                latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

                if self.use_prj:
                    latent_embd = self.prj(latent_embd)

            # Add eocot token to signal end of latent reasoning
            eocot_emb = embedding_layer(
                torch.tensor([[self.eocot_id]], device=device).expand(batch_size, -1)
            )

            outputs = self.model(
                inputs_embeds=eocot_emb,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values
            )
            past_key_values = outputs.past_key_values

            # Generate answer tokens
            # Use logits directly from eocot processing for first token (matches CODI repo)
            generated_ids = []
            next_token_logits = outputs.logits[:, -1, :self.model.config.vocab_size]
            first_iteration = True

            for _ in range(max_new_tokens):
                if not first_iteration:
                    next_emb = embedding_layer(next_token_id).unsqueeze(1)
                    outputs = self.model(
                        inputs_embeds=next_emb,
                        use_cache=True,
                        past_key_values=past_key_values
                    )
                    past_key_values = outputs.past_key_values
                    next_token_logits = outputs.logits[:, -1, :self.model.config.vocab_size]

                first_iteration = False
                next_token_id = self._sample_token(
                    next_token_logits, greedy, temperature, top_k, top_p
                )

                if batch_size == 1 and next_token_id[0].item() == self.tokenizer.eos_token_id:
                    break

                generated_ids.append(next_token_id.unsqueeze(1))

            # Construct output: input + [eot] + bocot + latent*N + eocot + generated
            output_parts = [input_ids]

            if self.use_eot:
                eot_tensor = torch.full((batch_size, 1), self.eot_id, dtype=torch.long, device=device)
                output_parts.append(eot_tensor)

            bocot_tensor = torch.full((batch_size, 1), self.bocot_id, dtype=torch.long, device=device)
            latent_placeholder = torch.full(
                (batch_size, num_iterations), self.latent_id, dtype=torch.long, device=device
            )
            eocot_tensor = torch.full((batch_size, 1), self.eocot_id, dtype=torch.long, device=device)

            output_parts.extend([bocot_tensor, latent_placeholder, eocot_tensor])
            if generated_ids:
                output_parts.append(torch.cat(generated_ids, dim=1))

            output_ids = torch.cat(output_parts, dim=1)

            return GenerateResult(output_ids=output_ids, mode="latent")

    def _autoregressive_generate(
        self,
        inputs_embeds: torch.Tensor,
        max_new_tokens: int,
        greedy: bool,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> torch.Tensor:
        """
        Standard autoregressive generation from initial embeddings.
        """
        device = inputs_embeds.device
        batch_size = inputs_embeds.shape[0]
        embedding_layer = self.get_embd(self.model, self.model_name)

        outputs = self.model(
            inputs_embeds=inputs_embeds,
            use_cache=True
        )
        past_key_values = outputs.past_key_values

        generated_ids = []
        next_token_logits = outputs.logits[:, -1, :self.model.config.vocab_size]
        next_token_id = self._sample_token(
            next_token_logits, greedy, temperature, top_k, top_p
        )

        for _ in range(max_new_tokens):
            if batch_size == 1 and next_token_id[0].item() == self.tokenizer.eos_token_id:
                break

            generated_ids.append(next_token_id.unsqueeze(1))

            next_embed = embedding_layer(next_token_id).unsqueeze(1)
            outputs = self.model(
                inputs_embeds=next_embed,
                use_cache=True,
                past_key_values=past_key_values
            )
            past_key_values = outputs.past_key_values

            next_token_logits = outputs.logits[:, -1, :self.model.config.vocab_size]
            next_token_id = self._sample_token(
                next_token_logits, greedy, temperature, top_k, top_p
            )

        if generated_ids:
            return torch.cat(generated_ids, dim=1)
        else:
            return torch.empty((batch_size, 0), dtype=torch.long, device=device)

    def _sample_token(
        self,
        logits: torch.Tensor,
        greedy: bool,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> torch.Tensor:
        """Sample next token from logits."""
        if greedy:
            return torch.argmax(logits, dim=-1)
        else:
            logits = logits / temperature
            if top_k > 1:
                top_k_values, _ = torch.topk(logits, top_k, dim=-1)
                min_top_k_value = top_k_values[:, -1].unsqueeze(-1)
                logits[logits < min_top_k_value] = float('-inf')

            if top_p < 1.0:
                sorted_logit, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logit, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                if sorted_indices_to_remove.any():
                    sorted_indices_to_remove = sorted_indices_to_remove.roll(1, dims=-1)
                    sorted_indices_to_remove[:, 0] = False
                for b in range(logits.size(0)):
                    logits[b, sorted_indices[b, sorted_indices_to_remove[b]]] = float('-inf')

            probs = torch.nn.functional.softmax(logits, dim=-1)
            return torch.multinomial(probs, num_samples=1).squeeze(-1)
