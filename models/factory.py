"""
Model factory for creating model instances by type.

This factory pattern allows experiments to create models without knowing
the specific implementation details, enabling easy addition of new models.
"""

from typing import Dict, Type, Any
from models.base import BaseModel


class ModelFactory:
    """
    Factory for creating model instances.

    Usage:
        >>> model = ModelFactory.create(
        ...     model_type="coconut",
        ...     model_path="checkpoints/gsm-coconut",
        ...     device="cuda"
        ... )
    """

    # Registry of available model types
    _registry: Dict[str, Type[BaseModel]] = {}

    @classmethod
    def create(
        cls,
        model_type: str,
        model_path: str,
        device: str = "cuda",
        **kwargs
    ) -> BaseModel:
        """
        Create and load a model instance.

        Args:
            model_type: Type of model ('cot', 'coconut', 'codi')
            model_path: Path to model checkpoint or identifier
            device: Device to load model on
            **kwargs: Model-specific parameters:
                - For Coconut: (parameters inferred from dataset)
                - For CODI: num_latent, use_prj, etc.

        Returns:
            Loaded model instance ready for inference

        Raises:
            ValueError: If model_type is not registered

        Example:
            >>> # Create a Coconut model
            >>> model = ModelFactory.create(
            ...     model_type="coconut",
            ...     model_path="checkpoints/gsm-coconut/checkpoint_33"
            ... )
            >>>
            >>> # Create a CODI model
            >>> model = ModelFactory.create(
            ...     model_type="codi",
            ...     model_path="checkpoints/codi-gsm8k",
            ...     num_latent=5,
            ...     use_prj=True
            ... )
        """
        if model_type not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(
                f"Unknown model type: '{model_type}'. "
                f"Available types: {available}. "
                f"Use ModelFactory.register() to add new model types."
            )

        model_class = cls._registry[model_type]

        # Create model instance
        model = model_class(model_path=model_path, device=device, **kwargs)

        # Load model weights and tokenizer
        model.load()

        return model

    @classmethod
    def register(cls, model_type: str, model_class: Type[BaseModel]) -> None:
        """
        Register a new model type.

        This allows easy extension with new models without modifying factory code.

        Args:
            model_type: Identifier for this model type (e.g., 'coconut', 'codi')
            model_class: Class implementing BaseModel interface

        Example:
            >>> from models.custom_model import MyCustomModel
            >>> ModelFactory.register('custom', MyCustomModel)
            >>> model = ModelFactory.create('custom', 'path/to/model')
        """
        if not issubclass(model_class, BaseModel):
            raise TypeError(
                f"Model class must inherit from BaseModel. "
                f"Got: {model_class}"
            )

        cls._registry[model_type] = model_class
        print(f"Registered model type: '{model_type}' -> {model_class.__name__}")

    @classmethod
    def list_models(cls) -> list:
        """
        List all registered model types.

        Returns:
            List of registered model type identifiers
        """
        return list(cls._registry.keys())

    @classmethod
    def is_registered(cls, model_type: str) -> bool:
        """
        Check if a model type is registered.

        Args:
            model_type: Model type identifier to check

        Returns:
            True if model type is registered, False otherwise
        """
        return model_type in cls._registry


# Auto-register models when they're imported
def _auto_register_models():
    """
    Automatically register model implementations.

    This is called when factory.py is imported, ensuring all standard
    models are available.
    """
    try:
        from models.cot.cot_model import CoTModel
        ModelFactory.register('cot', CoTModel)
    except ImportError:
        pass  # CoT model not yet implemented

    try:
        from models.coconut.coconut_model import CoconutModel
        ModelFactory.register('coconut', CoconutModel)
    except ImportError:
        pass  # Coconut model not yet implemented

    try:
        from models.codi.codi_model import CODIModel
        ModelFactory.register('codi', CODIModel)
    except ImportError as e:
        print(f"Failed to import CODI: {e}")
        pass  # CODI model not yet implemented

    try:
        from models.no_cot.no_cot_model import NoCotModel
        ModelFactory.register('no_cot', NoCotModel)
    except ImportError:
        pass  # No-CoT model not yet implemented

    try:
        from models.multimode_codi.multimode_codi_model import MultimodeCODIModel
        ModelFactory.register('multimode_codi', MultimodeCODIModel)
    except ImportError as e:
        print(f"Failed to import MultimodeCODI: {e}")
        pass  # Multimode CODI model not yet implemented

    try:
        from models.multimode_coconut.multimode_coconut_model import MultimodeCoconutModel
        ModelFactory.register('multimode_coconut', MultimodeCoconutModel)
    except ImportError as e:
        print(f"Failed to import MultimodeCoconut: {e}")
        pass  # Multimode Coconut model not yet implemented


# Register models on import
_auto_register_models()
