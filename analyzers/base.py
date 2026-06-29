"""
Unified analyzer for model-agnostic interpretability experiments.

This module provides a single analyzer that works with any model implementing
the BaseModel interface, replacing model-specific analyzers.
"""

from typing import Dict, List, Optional, Tuple
import torch

from models.base import BaseModel, ActivationCapture, ModelOutput


class UnifiedAnalyzer:
    """
    Model-agnostic analyzer for interpretability experiments.

    This analyzer works with any model implementing the BaseModel interface,
    providing common analysis operations like:
    - Activation capture during generation
    - Vocabulary projection
    - Top-k token prediction
    """

    def __init__(self, model: BaseModel):
        """
        Initialize unified analyzer.

        Args:
            model: Any model implementing BaseModel interface
        """
        self.model = model
        # Coconut uses multi-pass generation, so only keep last pass
        # Other models (CoT, CODI) need all passes
        keep_only_last = (model.model_type == "coconut")
        self.activation_capture = ActivationCapture(model, keep_only_last_pass=keep_only_last)

    def analyze_with_capture(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 64,
        layer_indices: Optional[List[int]] = None,
        **gen_kwargs
    ) -> Dict:
        """
        Run inference and capture activations.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            max_new_tokens: Maximum tokens to generate
            layer_indices: Which layers to capture (None = all)
            **gen_kwargs: Additional generation parameters

        Returns:
            Dictionary with:
                - 'output': ModelOutput from generation
                - 'activations': Dict[layer_idx -> tensor]
                - 'decoded': Decoded output text
        """
        # Register hooks
        self.activation_capture.register_hooks(layer_indices)

        try:
            # Run generation
            output = self.model.generate(
                input_ids,
                attention_mask,
                max_new_tokens,
                **gen_kwargs
            )

            # Get captured activations
            activations = self.activation_capture.get_activations(concatenate=True)

            # Decode output
            decoded = self.model.decode(output.output_ids)

            return {
                "output": output,
                "activations": activations,
                "decoded": decoded
            }

        finally:
            # Always remove hooks
            self.activation_capture.remove_hooks()

    def project_activations_to_vocab(
        self,
        activations: torch.Tensor,
        top_k: int = 10,
        return_probs: bool = True
    ) -> Dict:
        """
        Project activations to vocabulary space.

        Args:
            activations: Hidden states [batch, seq_len, hidden_dim]
            top_k: Number of top tokens to return
            return_probs: Whether to return probabilities (True) or just logits

        Returns:
            Dictionary with:
                - 'logits': [batch, seq_len, vocab_size]
                - 'top_k_indices': [batch, seq_len, top_k]
                - 'top_k_probs': [batch, seq_len, top_k] (if return_probs)
                - 'top_k_tokens': List of decoded token strings
        """
        # Project to vocab
        logits = self.model.project_to_vocab(activations)

        # Get probabilities if requested
        if return_probs:
            probs = torch.softmax(logits, dim=-1)
            top_k_probs, top_k_indices = torch.topk(probs, k=top_k, dim=-1)
        else:
            top_k_values, top_k_indices = torch.topk(logits, k=top_k, dim=-1)
            top_k_probs = top_k_values

        # Decode top-k tokens
        top_k_tokens = self._decode_top_k_batch(top_k_indices)

        result = {
            "logits": logits,
            "top_k_indices": top_k_indices,
            "top_k_probs": top_k_probs if return_probs else None,
            "top_k_tokens": top_k_tokens
        }

        return result

    def _decode_top_k_batch(self, top_k_indices: torch.Tensor) -> List[List[List[str]]]:
        """
        Decode top-k token indices to strings.

        Args:
            top_k_indices: [batch, seq_len, top_k]

        Returns:
            List[batch][seq_pos][top_k] of decoded token strings
        """
        batch_size, seq_len, top_k = top_k_indices.shape

        decoded_batch = []
        for batch_idx in range(batch_size):
            decoded_seq = []
            for pos_idx in range(seq_len):
                decoded_tokens = [
                    self.model.tokenizer.decode([idx.item()])
                    for idx in top_k_indices[batch_idx, pos_idx]
                ]
                decoded_seq.append(decoded_tokens)
            decoded_batch.append(decoded_seq)

        return decoded_batch

    def get_top_k_at_position(
        self,
        activations: torch.Tensor,
        position: int,
        top_k: int = 10
    ) -> Tuple[List[str], torch.Tensor]:
        """
        Get top-k predicted tokens at a specific sequence position.

        Args:
            activations: Hidden states [batch, seq_len, hidden_dim]
            position: Sequence position to analyze
            top_k: Number of top tokens

        Returns:
            Tuple of (token_strings, probabilities)
        """
        # Extract position
        hidden_at_pos = activations[:, position:position+1, :]

        # Project to vocab
        result = self.project_activations_to_vocab(hidden_at_pos, top_k=top_k)

        # Extract for first batch, first (only) position
        tokens = result['top_k_tokens'][0][0]
        probs = result['top_k_probs'][0, 0, :] if result['top_k_probs'] is not None else None

        return tokens, probs

    def compare_with_answer(
        self,
        activations: torch.Tensor,
        answer_text: str,
        top_k: int = 10
    ) -> Dict:
        """
        Check if answer appears in top-k predictions at each position.

        Args:
            activations: Hidden states [batch, seq_len, hidden_dim]
            answer_text: Ground truth answer to look for
            top_k: Number of top predictions to check

        Returns:
            Dictionary with:
                - 'first_appearance': First position where answer in top-k (or None)
                - 'positions': List of positions where answer appears
                - 'ranks': Rank of answer at each position (or None if not in top-k)
        """
        # Get answer token variants (with/without space)
        answer_ids = set()
        for variant in [answer_text, " " + answer_text]:
            tokens = self.model.tokenizer.encode(variant, add_special_tokens=False)
            if tokens:
                answer_ids.add(tokens[0])

        # Project all positions
        result = self.project_activations_to_vocab(activations, top_k=top_k)
        top_k_indices = result['top_k_indices']  # [batch, seq_len, top_k]

        # Check each position
        seq_len = top_k_indices.shape[1]
        positions = []
        ranks = []
        first_appearance = None

        for pos in range(seq_len):
            top_k_at_pos = top_k_indices[0, pos, :].tolist()

            # Check if any answer variant in top-k
            for rank, token_id in enumerate(top_k_at_pos):
                if token_id in answer_ids:
                    if first_appearance is None:
                        first_appearance = pos
                    positions.append(pos)
                    ranks.append(rank)
                    break

        return {
            "first_appearance": first_appearance,
            "positions": positions,
            "ranks": ranks,
            "answer_ids": list(answer_ids)
        }

    def clear_cache(self):
        """Clear activation cache."""
        self.activation_capture.clear()

    def __repr__(self) -> str:
        return f"UnifiedAnalyzer(model={self.model.model_type})"
