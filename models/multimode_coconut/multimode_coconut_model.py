"""
Multi-mode Coconut model implementation with BaseModel interface.

This is for models trained with run.py using multimode=True in the config.
Unlike multimode_codi, this does NOT use LoRA or projection layers.

Supports three inference modes:
- Direct: No reasoning, prompt + eot + eocot -> answer
- Verbalized: Standard autoregressive CoT, prompt + eot + bocot -> CoT + answer
- Latent: Hidden state replacement, prompt + eot + bocot + latent*N + eocot -> answer
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Optional

from models.base import BaseModel, ModelOutput, ModelComponents
from src_coconut_multimode.coconut import Coconut


class MultimodeCoconutModel(BaseModel):
    """
    Multi-mode Coconut model with three inference modes.

    This model is trained with run.py using multimode=True config.
    It uses the Coconut wrapper without LoRA/projection layers.

    Special tokens:
    - <|eot_id|>: End of turn (separates question from reasoning)
    - <|bocot|>: Beginning of chain-of-thought
    - <|eocot|>: End of chain-of-thought
    - <|latent|>: Latent reasoning token
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        model_id: str = "openai-community/gpt2",
        num_latent: int = 6,
        **kwargs
    ):
        """
        Initialize Multi-mode Coconut model.

        Args:
            model_path: Path to checkpoint (from run.py training)
            device: Device to load model on
            model_id: Base model identifier (e.g., "openai-community/gpt2" or
                      "meta-llama/Llama-3.2-1B-Instruct")
            num_latent: Number of latent tokens for latent mode
        """
        super().__init__(model_path, device, **kwargs)
        self.model_id = model_id
        self.num_latent = num_latent
        self._special_token_ids = {}

    def load(self) -> None:
        """Load Multi-mode Coconut model with special tokens."""
        print(f"Loading Multi-mode Coconut model from {self.model_id}...")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Determine if we should use eot token based on model type
        self.use_eot = "llama" in self.model_id.lower()

        if self.use_eot:
            # Llama: <|eot_id|> is already built-in, get it from vocab
            eot_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            if eot_id == self.tokenizer.unk_token_id:
                # Fallback: add it if not found (shouldn't happen for Llama-3.x)
                self.tokenizer.add_tokens("<|eot_id|>")
                eot_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        else:
            # GPT-2: No eot token
            eot_id = None

        # Add bocot, eocot, latent tokens (for all models)
        self.tokenizer.add_tokens("<|bocot|>")
        self.tokenizer.add_tokens("<|eocot|>")
        self.tokenizer.add_tokens("<|latent|>")

        # Get token IDs
        bocot_id = self.tokenizer.convert_tokens_to_ids("<|bocot|>")
        eocot_id = self.tokenizer.convert_tokens_to_ids("<|eocot|>")
        latent_id = self.tokenizer.convert_tokens_to_ids("<|latent|>")

        self._special_token_ids = {
            "eot": eot_id,
            "bocot": bocot_id,
            "eocot": eocot_id,
            "latent": latent_id,
        }

        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            attn_implementation="eager"  # Required for custom attention masks
        )
        base_model.resize_token_embeddings(len(self.tokenizer))

        # Wrap in Coconut with multimode tokens
        # For multimode, start_id=bocot_id and end_id=eocot_id
        self.model = Coconut(
            base_causallm=base_model,
            latent_token_id=latent_id,
            start_latent_id=bocot_id,  # bocot serves as start marker
            end_latent_id=eocot_id,    # eocot serves as end marker
            eos_token_id=self.tokenizer.eos_token_id,
            eot_token_id=eot_id,
            bocot_token_id=bocot_id,
            eocot_token_id=eocot_id,
        )

        # Load checkpoint
        if not self.model_path:
            raise ValueError("model_path is required for multimode_coconut")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.model_path}")

        print(f"Loading checkpoint from {self.model_path}...")
        saved_weights = torch.load(self.model_path, map_location=self.device)
        result = self.model.load_state_dict(saved_weights, strict=False)
        print(f"Loaded (missing: {len(result.missing_keys)}, unexpected: {len(result.unexpected_keys)})")

        # Move to device and set to eval
        self.model = self.model.to(self.device)
        self.model.eval()
        print("Multi-mode Coconut model loaded and ready")

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 64,
        mode: str = "latent",
        num_iterations: Optional[int] = None,
        synced_gpus: bool = False,
        output_embedding: bool = False,
        **kwargs
    ) -> ModelOutput:
        """
        Generate using specified mode.

        Args:
            input_ids: Question tokens (should include appropriate mode tokens)
            attention_mask: Attention mask
            max_new_tokens: Maximum tokens to generate
            mode: One of "direct", "verbalized", "latent"
            num_iterations: Number of latent iterations (for latent mode)
            synced_gpus: Whether to sync GPUs (for FSDP)
            output_embedding: Whether to return embeddings
            **kwargs: Additional generation parameters

        Returns:
            ModelOutput with generated tokens
        """
        if num_iterations is None:
            num_iterations = self.num_latent

        with torch.no_grad():
            result = self.model.generate(
                input_ids=input_ids.to(self.device),
                attention_mask=attention_mask.to(self.device),
                max_new_tokens=max_new_tokens,
                output_embedding=output_embedding,
                synced_gpus=synced_gpus,
                mode=mode,
                **kwargs
            )

        if output_embedding:
            output_ids, embeddings = result
            return ModelOutput(
                output_ids=output_ids,
                metadata={
                    "embeddings": embeddings,
                    "mode": mode,
                    "num_iterations": num_iterations,
                    "delimiter": "###",  # Same as coconut
                    **self._special_token_ids
                }
            )
        else:
            return ModelOutput(
                output_ids=result,
                metadata={
                    "mode": mode,
                    "num_iterations": num_iterations,
                    "delimiter": "###",
                    **self._special_token_ids
                }
            )

    def get_components(self) -> ModelComponents:
        """Get components from base model."""
        base_model = self.model.base_causallm

        # Handle both GPT2-style and Llama-style models
        if hasattr(base_model, 'model'):
            # Llama-style
            layers = list(base_model.model.layers)
            embedding = base_model.model.embed_tokens
            layer_norm = base_model.model.norm
        else:
            # GPT2-style
            layers = list(base_model.transformer.h)
            embedding = base_model.transformer.wte
            layer_norm = base_model.transformer.ln_f

        return ModelComponents(
            lm_head=base_model.lm_head,
            layer_norm=layer_norm,
            embedding=embedding,
            layers=layers
        )

    def prepare_input(
        self,
        question: str,
        mode: str = "latent",
        num_latents: int = 6,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare input for specified mode.

        Args:
            question: Question text
            mode: One of "direct", "verbalized", "latent"
            num_latents: Number of latent tokens (for latent mode)
            **kwargs: Additional parameters

        Returns:
            Dictionary with input_ids and attention_mask
        """
        # Tokenize question
        question_text = question if question.endswith("\n") else question + "\n"
        question_ids = self.tokenizer.encode(question_text, add_special_tokens=True)

        eot_id = self._special_token_ids["eot"]
        bocot_id = self._special_token_ids["bocot"]
        eocot_id = self._special_token_ids["eocot"]
        latent_id = self._special_token_ids["latent"]

        # Control tokens after prompt depend on use_eot
        # GPT-2: 1 control token (no eot)
        # Llama: 2 control tokens (eot + mode token)
        eot_tokens = [eot_id] if self.use_eot else []

        if mode == "direct":
            # Direct: question + [eot] + eocot
            tokens = question_ids + eot_tokens + [eocot_id]
        elif mode == "verbalized":
            # Verbalized: question + [eot] + bocot (model generates CoT + eocot + answer)
            tokens = question_ids + eot_tokens + [bocot_id]
        elif mode == "latent":
            # Latent: question + [eot] + bocot + latent*N + eocot
            tokens = question_ids + eot_tokens + [bocot_id] + [latent_id] * num_latents + [eocot_id]
        else:
            raise ValueError(f"Unknown mode: {mode}. Must be 'direct', 'verbalized', or 'latent'")

        input_ids = torch.tensor(tokens).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "mode": mode,
            "num_latents": num_latents if mode == "latent" else 0,
        }

    @property
    def model_type(self) -> str:
        """Return model type identifier."""
        return "multimode_coconut"

    @property
    def special_tokens(self) -> Dict[str, int]:
        """Return special token IDs."""
        return self._special_token_ids

    def find_reasoning_start(self, output_ids: torch.Tensor) -> int:
        """
        Find where reasoning starts (position of <|bocot|> token).

        Args:
            output_ids: Output token IDs

        Returns:
            Index of first <|bocot|> token, or 0 if not found
        """
        if output_ids.dim() > 1:
            output_ids = output_ids[0]

        bocot_id = self._special_token_ids.get('bocot')
        if bocot_id is None:
            return 0

        positions = (output_ids == bocot_id).nonzero(as_tuple=True)[0]
        if len(positions) > 0:
            return positions[0].item()

        return 0

    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = False) -> str:
        """
        Decode token IDs to text with special handling for multimode tokens.

        Args:
            token_ids: Token IDs to decode
            skip_special_tokens: Whether to skip special tokens

        Returns:
            Decoded text string
        """
        if token_ids.dim() > 1:
            token_ids = token_ids[0]

        token_list = token_ids.tolist()

        eot_id = self._special_token_ids.get("eot")
        bocot_id = self._special_token_ids.get("bocot")
        eocot_id = self._special_token_ids.get("eocot")
        latent_id = self._special_token_ids.get("latent")

        output_parts = []
        current_segment = []

        for token_id in token_list:
            if token_id == eot_id:
                if current_segment:
                    output_parts.append(self.tokenizer.decode(current_segment, skip_special_tokens=skip_special_tokens))
                    current_segment = []
                output_parts.append(' <|eot|>')

            elif token_id == bocot_id:
                if current_segment:
                    output_parts.append(self.tokenizer.decode(current_segment, skip_special_tokens=skip_special_tokens))
                    current_segment = []
                output_parts.append(' <|bocot|>')

            elif token_id == eocot_id:
                if current_segment:
                    output_parts.append(self.tokenizer.decode(current_segment, skip_special_tokens=skip_special_tokens))
                    current_segment = []
                output_parts.append(' <|eocot|>')

            elif token_id == latent_id:
                if current_segment:
                    output_parts.append(self.tokenizer.decode(current_segment, skip_special_tokens=skip_special_tokens))
                    current_segment = []
                output_parts.append(' <|latent|>')

            else:
                current_segment.append(token_id)

        if current_segment:
            output_parts.append(self.tokenizer.decode(current_segment, skip_special_tokens=skip_special_tokens))

        return ''.join(output_parts)
