"""
Base classes and interfaces for multi-model interpretability analysis.

This module provides the core abstractions that all models must implement,
enabling model-agnostic experiments and analysis tools.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
import torch
import torch.nn as nn
from transformers import PreTrainedTokenizer


@dataclass
class ModelOutput:
    """Standardized output from model generation."""
    output_ids: torch.Tensor  # Generated token IDs [batch_size, seq_len]
    hidden_states: Optional[List[torch.Tensor]] = None  # Per-layer hidden states
    logits: Optional[torch.Tensor] = None  # Final logits
    metadata: Optional[Dict[str, Any]] = None  # Model-specific info (delimiters, special tokens, etc.)

    def to(self, device):
        """Move tensors to device."""
        self.output_ids = self.output_ids.to(device)
        if self.hidden_states is not None:
            self.hidden_states = [h.to(device) if h is not None else None for h in self.hidden_states]
        if self.logits is not None:
            self.logits = self.logits.to(device)
        return self


@dataclass
class ModelComponents:
    """Access to internal model components for analysis."""
    lm_head: nn.Module  # Unembedding matrix (vocab projection)
    layer_norm: nn.Module  # Final layer normalization
    embedding: nn.Module  # Input embeddings
    layers: List[nn.Module]  # Transformer layers (for hooking)


class BaseModel(ABC):
    """
    Abstract base class for all models in interpretability analysis.

    All models (CoT, Coconut, CODI, etc.) must implement this interface
    to work with experiments and analyzers.
    """

    def __init__(self, model_path: str, device: str = "cuda", **kwargs):
        """
        Initialize base model.

        Args:
            model_path: Path to model checkpoint or model identifier
            device: Device to load model on ('cuda', 'cpu', 'mps')
            **kwargs: Model-specific parameters
        """
        self.model_path = model_path
        self.device = torch.device(device)
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.model: Optional[nn.Module] = None

    @abstractmethod
    def load(self) -> None:
        """
        Load model and tokenizer from checkpoint.

        This method should:
        1. Load the tokenizer
        2. Set up special tokens if needed
        3. Load the model weights
        4. Move model to self.device
        5. Set model to eval mode
        """
        pass

    @abstractmethod
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 64,
        **kwargs
    ) -> ModelOutput:
        """
        Generate output tokens.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            max_new_tokens: Maximum tokens to generate
            **kwargs: Model-specific generation parameters
                - do_sample: Whether to sample (default False)
                - temperature: Sampling temperature
                - top_k, top_p: Sampling parameters
                - output_hidden_states: Whether to return hidden states

        Returns:
            ModelOutput with generated tokens and optional hidden states/logits
        """
        pass

    @abstractmethod
    def get_components(self) -> ModelComponents:
        """
        Get internal model components for analysis.

        Returns:
            ModelComponents with lm_head, layer_norm, embedding, and layers
        """
        pass

    @abstractmethod
    def prepare_input(self, question: str, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Prepare input for model inference.

        Args:
            question: Question text
            **kwargs: Model-specific parameters
                - For Coconut: num_latents
                - For CODI: num_iterations

        Returns:
            Dictionary with at least:
                - 'input_ids': torch.Tensor
                - 'attention_mask': torch.Tensor
        """
        pass

    @property
    @abstractmethod
    def model_type(self) -> str:
        """
        Return model type identifier.

        Returns:
            One of: 'cot', 'coconut', 'codi', etc.
        """
        pass

    @property
    def special_tokens(self) -> Dict[str, int]:
        """
        Return model-specific special token IDs.

        Returns:
            Dictionary mapping token name to token ID.
            Empty dict if no special tokens.
        """
        return {}

    def project_to_vocab(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Project hidden states to vocabulary space.

        This is a common operation for all models:
        hidden_states -> layer_norm -> lm_head -> logits

        Args:
            hidden_states: [batch, seq_len, hidden_dim] or [seq_len, hidden_dim]

        Returns:
            logits: [batch, seq_len, vocab_size] or [seq_len, vocab_size]
        """
        components = self.get_components()

        # Ensure batch dimension
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        with torch.no_grad():
            # Move to same device as model
            hidden_states = hidden_states.to(next(components.layer_norm.parameters()).device)

            # Apply final layer norm and projection
            normalized = components.layer_norm(hidden_states)
            logits = components.lm_head(normalized)

            if squeeze_output:
                logits = logits.squeeze(0)

        return logits

    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = False) -> str:
        """
        Decode token IDs to text.

        Args:
            token_ids: Token IDs to decode (can be 1D or 2D)
            skip_special_tokens: Whether to skip special tokens in output

        Returns:
            Decoded text string
        """
        if token_ids.dim() > 1:
            token_ids = token_ids[0]  # Take first batch
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def find_reasoning_start(self, output_ids: torch.Tensor) -> int:
        """
        Find the position where reasoning/answer generation starts in output.

        This is useful for experiments that want to analyze only the reasoning
        portion of the output, excluding the input question.

        Default implementation returns 0 (analyze from beginning).
        Models should override this to find model-specific reasoning markers.

        Args:
            output_ids: Output token IDs [batch_size, seq_len] or [seq_len]

        Returns:
            Index where reasoning starts (0-based)

        Examples:
            - CoT: Position after input question (handled separately via input_length)
            - Coconut: Position of <|start-latent|> token
            - CODI: Position of <|bot|> token
        """
        return 0

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_path='{self.model_path}', device='{self.device}')"


class ActivationCapture:
    """
    Unified interface for capturing activations during inference.

    Supports two modes:
    1. On-the-fly: Register hooks and capture during generation
    2. Pre-saved: Load activations from disk
    """

    def __init__(self, model: BaseModel, keep_only_last_pass: bool = False):
        """
        Initialize activation capture.

        Args:
            model: BaseModel instance to capture activations from
            keep_only_last_pass: If True, only keep the last forward pass (for Coconut).
                                 If False, keep all passes (for CoT, CODI).
        """
        self.model = model
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        self.activations: Dict[int, List[torch.Tensor]] = {}
        self.num_layers: Optional[int] = None
        self.keep_only_last_pass = keep_only_last_pass

    def register_hooks(
        self,
        layer_indices: Optional[List[int]] = None,
    ) -> None:
        """
        Register forward hooks on specified layers.

        Args:
            layer_indices: Which layers to hook (None = all layers)
        """
        components = self.model.get_components()
        self.num_layers = len(components.layers)

        if layer_indices is None:
            layer_indices = list(range(self.num_layers))

        # Initialize activation storage
        for idx in layer_indices:
            self.activations[idx] = []

        # Register hooks
        for idx in layer_indices:
            hook = components.layers[idx].register_forward_hook(
                self._create_hook_fn(idx)
            )
            self.hooks.append(hook)

    def _create_hook_fn(self, layer_idx: int) -> Callable:
        """Create hook function for a specific layer."""
        def hook(module, input, output):
            # Extract hidden states from output
            # Output can be tuple (hidden_states,) or just hidden_states
            hidden = output[0] if isinstance(output, tuple) else output

            if self.keep_only_last_pass:
                # For multi-pass models like Coconut, only keep the last forward pass
                # The last pass contains all hidden states due to KV caching
                if len(self.activations[layer_idx]) == 0:
                    self.activations[layer_idx].append(hidden.detach().cpu())
                else:
                    # Overwrite the last one (keep only most recent forward pass)
                    self.activations[layer_idx][-1] = hidden.detach().cpu()
            else:
                # For standard models (CoT, CODI), keep all forward passes
                self.activations[layer_idx].append(hidden.detach().cpu())

        return hook

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def get_activations(self, concatenate: bool = True) -> Dict[int, torch.Tensor]:
        """
        Get captured activations.

        For models with multiple forward passes (like Coconut), this will
        concatenate activations across passes if concatenate=True.

        Args:
            concatenate: Whether to concatenate multiple forward passes

        Returns:
            Dictionary mapping layer_idx -> activations tensor
            If concatenate=True: [batch, total_seq_len, hidden_dim]
            If concatenate=False: List of [batch, seq_len, hidden_dim]
        """
        if not concatenate:
            return self.activations

        result = {}
        for layer_idx, acts in self.activations.items():
            if len(acts) == 0:
                result[layer_idx] = None
            elif len(acts) == 1:
                result[layer_idx] = acts[0]
            else:
                # Concatenate along sequence dimension
                result[layer_idx] = torch.cat(acts, dim=1)

        return result

    def clear(self) -> None:
        """Clear stored activations."""
        self.activations = {idx: [] for idx in self.activations.keys()}

    @staticmethod
    def load_from_disk(path: str, device: str = 'cpu') -> Dict[int, torch.Tensor]:
        """
        Load pre-saved activations from disk.

        Args:
            path: Path to saved activations (.pt file)
            device: Device to load tensors to

        Returns:
            Dictionary mapping layer_idx -> activations tensor
        """
        data = torch.load(path, map_location=device)

        # Handle different save formats
        if 'activations' in data:
            return data['activations']
        else:
            # Assume data is directly the activations dict
            return data

    def __enter__(self):
        """Context manager support."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensure hooks are removed on exit."""
        self.remove_hooks()
