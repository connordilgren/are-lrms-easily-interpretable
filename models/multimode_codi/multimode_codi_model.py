"""
Multi-mode CODI model implementation with BaseModel interface.

Supports three inference modes:
- Direct: No reasoning, direct answer generation
- Verbalized: Standard autoregressive CoT generation
- Latent: Hidden state replacement (iterative latent reasoning)
"""

import torch
from typing import Dict, Optional

from models.base import BaseModel, ModelOutput, ModelComponents
from models.multimode_codi.multimode_codi_wrapper import MultimodeCODIInference


class MultimodeCODIModel(BaseModel):
    """
    Multi-mode CODI model with three inference modes.

    This model can operate in:
    - direct: prompt + eot + eocot → answer (no reasoning)
    - verbalized: prompt + eot + bocot → CoT + eocot + answer
    - latent: prompt + eot + bocot → [latent iterations] → eocot → answer
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
        Initialize Multi-mode CODI model.

        Args:
            model_path: Path to checkpoint (local or HuggingFace ID)
            device: Device to load model on
            num_latent: Number of latent iterations for latent mode
            use_prj: Whether to use projection layer
            prj_dim: Projection layer dimension
            use_lora: Whether to use LoRA
            lora_r: LoRA rank
            lora_alpha: LoRA alpha
            checkpoint_path: Path to local checkpoint (if different from model_path)
            model_id: Base model ID (e.g., "gpt2" or "meta-llama/Llama-3.2-1B-Instruct")
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
        """Load Multi-mode CODI model for inference."""
        print(f"Loading Multi-mode CODI model...")

        # Create inference model
        self.model = MultimodeCODIInference(
            model_name_or_path=self.model_id,
            num_latent=self.num_latent,
            use_prj=self.use_prj,
            prj_dim=self.prj_dim,
            use_lora=self.use_lora,
            lora_r=self.lora_r,
            lora_alpha=self.lora_alpha,
            device=self.device.type
        )

        # Load checkpoint
        print(f"Loading checkpoint from {self.checkpoint_path}...")
        self.model.load_checkpoint(self.checkpoint_path)

        # Move to device
        self.model = self.model.to(self.device)
        self.model.eval()

        # Get tokenizer from model
        self.tokenizer = self.model.tokenizer

        # Store special token IDs
        self._special_token_ids = {
            "eot": self.model.eot_id,
            "bocot": self.model.bocot_id,
            "eocot": self.model.eocot_id,
            "pad": self.model.pad_token_id,
            "latent": self.model.latent_id
        }

        print("Multi-mode CODI model loaded and ready")

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 64,
        mode: str = "latent",
        num_iterations: Optional[int] = None,
        do_sample: bool = False,
        temperature: float = 1.0,
        **kwargs
    ) -> ModelOutput:
        """
        Generate using specified mode.

        Args:
            input_ids: Question tokens
            attention_mask: Attention mask
            max_new_tokens: Maximum tokens to generate
            mode: One of "direct", "verbalized", "latent"
            num_iterations: Number of latent iterations (for latent mode)
            do_sample: Whether to sample or use greedy decoding
            temperature: Sampling temperature
            **kwargs: Additional generation parameters

        Returns:
            ModelOutput with generated tokens
        """
        if num_iterations is None:
            num_iterations = self.num_latent

        with torch.no_grad():
            greedy = not do_sample

            result = self.model.generate(
                input_ids=input_ids.to(self.device),
                attention_mask=attention_mask.to(self.device),
                mode=mode,
                max_new_tokens=max_new_tokens,
                num_iterations=num_iterations,
                greedy=greedy,
                temperature=temperature,
                **kwargs
            )

        return ModelOutput(
            output_ids=result.output_ids,
            metadata={
                "mode": result.mode,
                "num_iterations": num_iterations,
                "delimiter": "The answer is:",
                "skip_special_tokens": False,
                **self._special_token_ids
            }
        )

    def get_components(self) -> ModelComponents:
        """Get components from base model."""
        base_model = self.model.model

        # Handle PEFT wrapper
        if hasattr(base_model, 'base_model'):
            base_model = base_model.base_model.model

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

    def prepare_input(self, question: str, mode: str = "latent", **kwargs) -> Dict[str, torch.Tensor]:
        """
        Prepare input for specified mode.

        Note: The generate method adds eot_id, bocot_id, etc. internally,
        so we just tokenize the question here.

        Args:
            question: Question text
            mode: Inference mode (for metadata)
            **kwargs: Additional parameters

        Returns:
            Dictionary with input_ids and attention_mask
        """
        encoded = self.tokenizer(
            question,
            return_tensors="pt",
            add_special_tokens=True
        )

        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "mode": mode
        }

    @property
    def model_type(self) -> str:
        """Return model type identifier."""
        return "multimode_codi"

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
        Decode token IDs to text with special handling for multi-mode tokens.

        For latent mode, the sequence is:
        - Llama: input + eot + bocot + bocot*N + eocot + answer
        - GPT-2: input + bocot + bocot*N + eocot + answer (no eot token)

        The first bocot (after eot for Llama, or first overall for GPT-2) is the
        actual <|bocot|>, subsequent ones are latent placeholders shown as <|latent|>.

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

        output_parts = []
        current_segment = []
        seen_eot = False
        seen_first_bocot = False

        for token_id in token_list:
            if token_id == eot_id:
                if current_segment:
                    output_parts.append(self.tokenizer.decode(current_segment, skip_special_tokens=skip_special_tokens))
                    current_segment = []
                output_parts.append(' <|eot|>')
                seen_eot = True

            elif token_id == bocot_id:
                if current_segment:
                    output_parts.append(self.tokenizer.decode(current_segment, skip_special_tokens=skip_special_tokens))
                    current_segment = []
                # First bocot after eot is the actual bocot marker
                # Subsequent ones are latent placeholders
                # For models without eot (GPT-2), the first bocot is the marker
                # For models with eot (Llama), require seeing eot first
                if (eot_id is None or seen_eot) and not seen_first_bocot:
                    output_parts.append(' <|bocot|>')
                    seen_first_bocot = True
                else:
                    output_parts.append(' <|latent|>')

            elif token_id == eocot_id:
                if current_segment:
                    output_parts.append(self.tokenizer.decode(current_segment, skip_special_tokens=skip_special_tokens))
                    current_segment = []
                output_parts.append(' <|eocot|>')

            else:
                current_segment.append(token_id)

        if current_segment:
            output_parts.append(self.tokenizer.decode(current_segment, skip_special_tokens=skip_special_tokens))

        return ''.join(output_parts)
