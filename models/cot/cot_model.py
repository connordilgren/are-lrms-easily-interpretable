"""
Chain-of-Thought (CoT) model implementation.

Standard transformer model fine-tuned on chain-of-thought reasoning.
Uses HuggingFace transformers AutoModelForCausalLM.
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict

from models.base import BaseModel, ModelOutput, ModelComponents


class CoTModel(BaseModel):
    """
    Standard Chain-of-Thought model.

    This is a wrapper around HuggingFace's AutoModelForCausalLM,
    providing a unified interface for interpretability analysis.
    """

    def __init__(self, model_path: str, device: str = "cuda", model_id: str = "openai-community/gpt2", **kwargs):
        """
        Initialize CoT model.

        Args:
            model_path: Path to model checkpoint file
            device: Device to load model on
            model_id: Base model identifier for tokenizer (default: gpt2)
        """
        super().__init__(model_path, device, **kwargs)
        self.model_id = model_id

    def load(self) -> None:
        """Load CoT model and tokenizer."""
        print(f"Loading CoT model from {self.model_id}...")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id)

        # Load fine-tuned checkpoint if provided
        if self.model_path and os.path.exists(self.model_path):
            print(f"Loading CoT checkpoint from {self.model_path}...")
            saved_weights = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(saved_weights, strict=False)
        else:
            print(f"Warning: No checkpoint found at {self.model_path}, using base model")

        # Move to device and set to eval
        self.model = self.model.to(self.device)
        self.model.eval()
        print("CoT model loaded and ready")

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 64,
        output_hidden_states: bool = False,
        do_sample: bool = False,
        **kwargs
    ) -> ModelOutput:
        """
        Generate using standard HuggingFace generate().

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            max_new_tokens: Maximum tokens to generate
            output_hidden_states: Whether to return hidden states
            do_sample: Whether to use sampling (False = greedy)
            **kwargs: Additional generation parameters

        Returns:
            ModelOutput with generated tokens and optional hidden states
        """
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids.to(self.device),
                attention_mask=attention_mask.to(self.device),
                max_new_tokens=max_new_tokens,
                output_hidden_states=output_hidden_states,
                return_dict_in_generate=True,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )

        return ModelOutput(
            output_ids=outputs.sequences,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            metadata={"delimiter": "###"}  # CoT uses ### to separate reasoning from answer
        )

    def get_components(self) -> ModelComponents:
        """Get model components for analysis."""
        # Handle both GPT2-style and Llama-style models
        if hasattr(self.model, 'model'):
            # Llama-style: model.model.layers, model.model.embed_tokens, model.model.norm
            layers = list(self.model.model.layers)
            embedding = self.model.model.embed_tokens
            layer_norm = self.model.model.norm
        else:
            # GPT2-style: model.transformer.h, model.transformer.wte, model.transformer.ln_f
            layers = list(self.model.transformer.h)
            embedding = self.model.transformer.wte
            layer_norm = self.model.transformer.ln_f

        return ModelComponents(
            lm_head=self.model.lm_head,
            layer_norm=layer_norm,
            embedding=embedding,
            layers=layers
        )

    def prepare_input(self, question: str, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Prepare input for CoT model.

        Args:
            question: Question text
            **kwargs: Additional parameters (unused for CoT)

        Returns:
            Dictionary with input_ids and attention_mask
        """
        # Add newline after question (standard CoT format)
        text = question if question.endswith("\n") else question + "\n"

        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=True
        )

        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"]
        }

    @property
    def model_type(self) -> str:
        """Return model type identifier."""
        return "cot"

    def find_reasoning_start(self, output_ids: torch.Tensor) -> int:
        """
        Find where reasoning starts in CoT output.

        For CoT models, reasoning starts immediately after the input question.
        Since this method doesn't have access to the input length, it returns 0.
        Experiments should track the input length separately and use that instead.

        Args:
            output_ids: Output token IDs [batch_size, seq_len] or [seq_len]

        Returns:
            0 (CoT requires input_length to be tracked separately)
        """
        # CoT doesn't have special reasoning marker tokens
        # The experiment needs to track input_length separately
        return 0
