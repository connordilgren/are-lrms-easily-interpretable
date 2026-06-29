"""
CODI model implementation with continuous latent embeddings.

CODI uses continuous latent embeddings (not explicit tokens) to compress
chain-of-thought reasoning into a continuous representation.
"""

import os
import torch
from typing import Dict, Optional

from models.base import BaseModel, ModelOutput, ModelComponents
from models.codi.codi_wrapper import CODIInference, create_inference_model


class CODIModel(BaseModel):
    """
    CODI model with continuous latent embeddings.

    Unlike Coconut (which uses explicit latent tokens), CODI uses
    continuous latent embeddings that bypass tokenization entirely.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        num_latent: int = 6,
        use_prj: bool = True,
        prj_dim: Optional[int] = None,
        use_lora: bool = True,
        lora_r: int = 128,
        lora_alpha: int = 32,
        checkpoint_path: Optional[str] = None,
        model_id: str = "gpt2",
        **kwargs
    ):
        """
        Initialize CODI model.

        Args:
            model_path: Path to CODI checkpoint on HuggingFace (e.g., zen-E/CODI-gpt2)
            device: Device to load model on
            num_latent: Number of latent iterations
            use_prj: Whether to use projection layer
            prj_dim: Projection layer dimension (None = use model's hidden size)
            use_lora: Whether to use LoRA
            lora_r: LoRA rank
            lora_alpha: LoRA alpha
            checkpoint_path: Path to local checkpoint (if different from model_path)
            model_id: Base model ID (e.g., "gpt2" or "meta-llama/Llama-3.2-1B")
        """
        super().__init__(model_id, device, **kwargs)
        self.model_id = model_id
        self.num_latent = num_latent
        self.use_prj = use_prj
        self.prj_dim = prj_dim
        self.use_lora = use_lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.checkpoint_path = checkpoint_path or model_path
        self._special_token_ids = {}

    def load(self) -> None:
        """Load CODI model for inference."""
        print(f"Loading CODI model...")

        # Create inference model (loads base model + applies LoRA)
        self.model = create_inference_model(
            model_path=self.model_id,
            num_latent=self.num_latent,
            use_prj=self.use_prj,
            prj_dim=self.prj_dim,
            use_lora=self.use_lora,
            lora_r=self.lora_r,
            lora_alpha=self.lora_alpha,
            device=self.device.type
        )

        # Load checkpoint (trained weights)
        print(f"Loading CODI checkpoint from {self.checkpoint_path}...")
        self.model.load_checkpoint(self.checkpoint_path)

        # Get tokenizer from model
        self.tokenizer = self.model.tokenizer

        # Store special token IDs
        self._special_token_ids = {
            "bot": self.model.bot_id,
            "eot": self.model.eot_id,
            "pad": self.model.pad_token_id
        }

        print("CODI model loaded and ready")

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 64,
        num_iterations: Optional[int] = None,
        do_sample: bool = False,
        temperature: float = 1.0,
        **kwargs
    ) -> ModelOutput:
        """
        Generate using CODI's iterative latent generation.

        Args:
            input_ids: Question tokens
            attention_mask: Attention mask
            max_new_tokens: Maximum tokens to generate
            num_iterations: Number of latent iterations (defaults to model's num_latent)
            do_sample: Whether to sample or use greedy decoding
            temperature: Sampling temperature
            **kwargs: Additional generation parameters

        Returns:
            ModelOutput with generated tokens
        """
        if num_iterations is None:
            num_iterations = self.num_latent

        with torch.no_grad():
            # Convert do_sample to greedy (original CODI parameter)
            greedy = not do_sample

            output_ids = self.model.generate(
                input_ids=input_ids.to(self.device),
                attention_mask=attention_mask.to(self.device),
                max_new_tokens=max_new_tokens,
                num_iterations=num_iterations,
                greedy=greedy,
                temperature=temperature,
                **kwargs
            )

        return ModelOutput(
            output_ids=output_ids,
            metadata={
                "num_iterations": num_iterations,
                "delimiter": "The answer is:",  # CODI uses this format
                "skip_special_tokens": False,  # Show bot/eot markers
                **self._special_token_ids
            }
        )

    def get_components(self) -> ModelComponents:
        """Get components from CODI's base model."""
        base_model = self.model.codi

        # Handle PEFT wrapper (get actual base model)
        if hasattr(base_model, 'base_model'):
            base_model = base_model.base_model.model  # PEFT wraps it twice

        # Handle different model architectures
        if hasattr(base_model, 'transformer'):
            # GPT2-style
            layers = list(base_model.transformer.h)
            embedding = base_model.transformer.wte
            layer_norm = base_model.transformer.ln_f
        elif hasattr(base_model, 'model') and hasattr(base_model.model, 'layers'):
            # Llama-style
            layers = list(base_model.model.layers)
            embedding = base_model.model.embed_tokens
            layer_norm = base_model.model.norm
        else:
            # Pythia-style
            layers = list(base_model.gpt_neox.layers)
            embedding = base_model.gpt_neox.embed_in
            layer_norm = base_model.gpt_neox.final_layer_norm

        return ModelComponents(
            lm_head=base_model.lm_head,
            layer_norm=layer_norm,
            embedding=embedding,
            layers=layers
        )

    def prepare_input(self, question: str, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Prepare input for CODI.

        CODI needs the question followed by BOT (Beginning of Thought) token.
        This signals the model to start latent reasoning.

        Args:
            question: Question text
            **kwargs: Additional parameters (unused)

        Returns:
            Dictionary with input_ids and attention_mask
        """
        encoded = self.tokenizer(
            question,
            return_tensors="pt",
            add_special_tokens=True
        )

        # Append BOT token to signal start of latent reasoning
        bot_tensor = torch.tensor([[self._special_token_ids["bot"]]], dtype=torch.long)
        input_ids = torch.cat([encoded["input_ids"], bot_tensor], dim=1)
        attention_mask = torch.cat([encoded["attention_mask"], torch.ones_like(bot_tensor)], dim=1)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }

    @property
    def model_type(self) -> str:
        """Return model type identifier."""
        return "codi"

    @property
    def special_tokens(self) -> Dict[str, int]:
        """Return CODI special token IDs."""
        return self._special_token_ids

    def find_reasoning_start(self, output_ids: torch.Tensor) -> int:
        """
        Find where reasoning starts in CODI output (position of <|bot|> token).

        Args:
            output_ids: Output token IDs [batch_size, seq_len] or [seq_len]

        Returns:
            Index of first <|bot|> token, or 0 if not found
        """
        # Ensure we're working with the first batch
        if output_ids.dim() > 1:
            output_ids = output_ids[0]

        bot_id = self._special_token_ids.get('bot')
        if bot_id is None:
            return 0

        # Find first occurrence of bot token
        positions = (output_ids == bot_id).nonzero(as_tuple=True)[0]
        if len(positions) > 0:
            return positions[0].item()

        return 0

    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = False) -> str:
        """
        Decode token IDs to text with special handling for CODI visualization.

        CODI uses model-specific token IDs (bot_id, eot_id) that aren't in the
        tokenizer's vocabulary. We manually replace these with readable strings.

        Args:
            token_ids: Token IDs to decode
            skip_special_tokens: Whether to skip special tokens in output

        Returns:
            Decoded text string
        """
        if token_ids.dim() > 1:
            token_ids = token_ids[0]  # Take first batch

        # Convert to list for manipulation
        token_list = token_ids.tolist()

        # Build output by decoding segments and inserting special token strings
        bot_id = self._special_token_ids["bot"]
        eot_id = self._special_token_ids["eot"]

        # Track whether we've seen the first bot token (for latent visualization)
        seen_first_bot = False
        output_parts = []
        current_segment = []

        for token_id in token_list:
            if token_id == bot_id:
                # Decode current segment
                if current_segment:
                    output_parts.append(self.tokenizer.decode(current_segment, skip_special_tokens=skip_special_tokens))
                    current_segment = []

                # Add bot or latent marker
                if not seen_first_bot:
                    output_parts.append(' <|bot|>')
                    seen_first_bot = True
                else:
                    output_parts.append(' <|latent|>')  # Subsequent bots are latent iterations

            elif token_id == eot_id:
                # Decode current segment
                if current_segment:
                    output_parts.append(self.tokenizer.decode(current_segment, skip_special_tokens=skip_special_tokens))
                    current_segment = []

                output_parts.append(' <|eot|>')

            else:
                # Regular token - add to current segment
                current_segment.append(token_id)

        # Decode any remaining tokens
        if current_segment:
            output_parts.append(self.tokenizer.decode(current_segment, skip_special_tokens=skip_special_tokens))

        return ''.join(output_parts)
