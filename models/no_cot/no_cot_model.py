"""
No Chain-of-Thought (No-CoT) model implementation.

Standard transformer model fine-tuned to directly output answers without
chain-of-thought reasoning. Uses HuggingFace transformers AutoModelForCausalLM.
"""

from models.cot.cot_model import CoTModel


class NoCotModel(CoTModel):
    """
    No Chain-of-Thought model.

    This is identical to CoTModel but with a different model_type identifier.
    The model outputs answers directly using ### delimiter, just without
    any chain-of-thought reasoning steps.
    """

    @property
    def model_type(self) -> str:
        """Return model type identifier."""
        return "no_cot"
