"""
Coconut model implementation with explicit latent tokens.

Coconut uses special latent tokens that are physically present in the sequence
during inference, enabling multi-pass reasoning.
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict

from models.base import BaseModel, ModelOutput, ModelComponents
from models.coconut.coconut_wrapper import Coconut


class CoconutModel(BaseModel):
    """
    Coconut model with explicit latent tokens.

    Coconut wraps a base CausalLM and adds latent token support for
    continuous reasoning. Latent tokens are explicit in the sequence.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        model_id: str = "openai-community/gpt2",
        **kwargs
    ):
        """
        Initialize Coconut model.

        Args:
            model_path: Path to Coconut checkpoint
            device: Device to load model on
            model_id: Base model identifier (default: gpt2)
        """
        super().__init__(model_path, device, **kwargs)
        self.model_id = model_id
        self._special_token_ids = {}

    def load(self) -> None:
        """Load Coconut model with special tokens."""
        print(f"Loading Coconut model from {self.model_id}...")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Add special tokens
        self.tokenizer.add_tokens("<|start-latent|>")
        self.tokenizer.add_tokens("<|end-latent|>")
        self.tokenizer.add_tokens("<|latent|>")

        # Get token IDs
        self._special_token_ids = {
            "latent": self.tokenizer.convert_tokens_to_ids("<|latent|>"),
            "start_latent": self.tokenizer.convert_tokens_to_ids("<|start-latent|>"),
            "end_latent": self.tokenizer.convert_tokens_to_ids("<|end-latent|>")
        }

        # Load base model with eager attention (required for ablation analysis)
        # SDPA doesn't support custom 4D attention masks properly
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            attn_implementation="eager"
        )
        base_model.resize_token_embeddings(len(self.tokenizer))

        # Get eot_token_id for Llama models (returns None for GPT-2)
        eot_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if eot_token_id == self.tokenizer.unk_token_id:
            eot_token_id = None  # Token doesn't exist in vocab

        # Wrap in Coconut
        self.model = Coconut(
            base_causallm=base_model,
            latent_token_id=self._special_token_ids["latent"],
            start_latent_id=self._special_token_ids["start_latent"],
            end_latent_id=self._special_token_ids["end_latent"],
            eos_token_id=self.tokenizer.eos_token_id,
            eot_token_id=eot_token_id
        )

        # Load checkpoint if provided
        if self.model_path and os.path.exists(self.model_path):
            print(f"Loading Coconut checkpoint from {self.model_path}...")
            saved_weights = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(saved_weights, strict=False)
        else:
            print(f"Warning: No checkpoint found at {self.model_path}, using base model")

        # Move to device and set to eval
        self.model = self.model.to(self.device)
        self.model.eval()
        print("Coconut model loaded and ready")

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 64,
        output_embedding: bool = False,
        synced_gpus: bool = False,
        **kwargs
    ) -> ModelOutput:
        """
        Generate using Coconut's custom generate().

        Args:
            input_ids: Input token IDs (must include latent tokens)
            attention_mask: Attention mask
            max_new_tokens: Maximum tokens to generate
            output_embedding: Whether to return embeddings (for analysis)
            synced_gpus: Whether to sync GPUs (for FSDP)
            **kwargs: Additional generation parameters

        Returns:
            ModelOutput with generated tokens
        """
        with torch.no_grad():
            result = self.model.generate(
                input_ids=input_ids.to(self.device),
                attention_mask=attention_mask.to(self.device),
                max_new_tokens=max_new_tokens,
                output_embedding=output_embedding,
                synced_gpus=synced_gpus,
                **kwargs
            )

        if output_embedding:
            output_ids, embeddings = result
            return ModelOutput(
                output_ids=output_ids,
                metadata={
                    "embeddings": embeddings,
                    "delimiter": "###",
                    **self._special_token_ids
                }
            )
        else:
            return ModelOutput(
                output_ids=result,
                metadata={
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
        num_latents: int = 6,
        include_latent_markers: bool = True,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare input with or without latent tokens.

        Args:
            question: Question text
            num_latents: Number of latent tokens to include (ignored if include_latent_markers=False)
            include_latent_markers: Whether to include <|start-latent|>, <|latent|>, <|end-latent|> tokens.
                If False, formats input like no-CoT: just [question] without any latent markers.
                Used for ablation experiments to test if latent markers are doing meaningful work.
            **kwargs: Additional parameters

        Returns:
            Dictionary with input_ids and attention_mask
        """
        # Tokenize question
        question_text = question if question.endswith("\n") else question + "\n"
        question_ids = self.tokenizer.encode(question_text, add_special_tokens=True)

        if include_latent_markers:
            # Standard Coconut format: question + <|start-latent|> + <|latent|>*N + <|end-latent|>
            tokens = (
                question_ids +
                [self._special_token_ids["start_latent"]] +
                [self._special_token_ids["latent"]] * num_latents +
                [self._special_token_ids["end_latent"]]
            )
        else:
            # No-latent format: just the question (like no-CoT)
            tokens = question_ids

        input_ids = torch.tensor(tokens).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "num_latents": num_latents if include_latent_markers else 0,
            "include_latent_markers": include_latent_markers
        }

    @property
    def model_type(self) -> str:
        """Return model type identifier."""
        return "coconut"

    @property
    def special_tokens(self) -> Dict[str, int]:
        """Return Coconut special token IDs."""
        return self._special_token_ids

    def find_reasoning_start(self, output_ids: torch.Tensor) -> int:
        """
        Find where reasoning starts in Coconut output (position of <|start-latent|> token).

        Args:
            output_ids: Output token IDs [batch_size, seq_len] or [seq_len]

        Returns:
            Index of <|start-latent|> token, or 0 if not found
        """
        # Ensure we're working with the first batch
        if output_ids.dim() > 1:
            output_ids = output_ids[0]

        start_latent_id = self._special_token_ids.get('start_latent')
        if start_latent_id is None:
            # Fallback: try to get from tokenizer
            try:
                start_latent_id = self.tokenizer.convert_tokens_to_ids('<|start-latent|>')
            except:
                return 0

        # Find first occurrence of start-latent token
        positions = (output_ids == start_latent_id).nonzero(as_tuple=True)[0]
        if len(positions) > 0:
            return positions[0].item()

        return 0
