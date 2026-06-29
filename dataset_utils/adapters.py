"""
Dataset adapters for model-specific input preparation.

Each model may require different input formats (e.g., Coconut needs latent tokens,
CODI needs specific iteration counts). Adapters handle these transformations.
"""

from typing import List, Dict, Any, Optional
import torch


class DatasetAdapter:
    """
    Adapter to convert datasets to model-specific formats.

    This class provides static methods to adapt dataset samples for
    different model types.
    """

    @staticmethod
    def adapt_batch_for_model(
        dataset,
        model,
        indices: Optional[List[int]] = None,
        **model_kwargs
    ) -> List[Dict[str, Any]]:
        """
        Adapt a batch of dataset samples for a specific model.

        Args:
            dataset: Dataset loaded from dataset_utils.base.load_dataset()
            model: BaseModel instance
            indices: Which samples to adapt (None = all)
            **model_kwargs: Model-specific parameters
                - For Coconut: num_latents
                - For CODI: num_iterations

        Returns:
            List of prepared inputs ready for model.generate()
        """
        if indices is None:
            indices = range(len(dataset))

        model_type = model.model_type

        if model_type == "cot":
            return DatasetAdapter._adapt_cot(dataset, model, indices)
        elif model_type == "coconut":
            return DatasetAdapter._adapt_coconut(
                dataset, model, indices, **model_kwargs
            )
        elif model_type == "codi":
            return DatasetAdapter._adapt_codi(
                dataset, model, indices, **model_kwargs
            )
        elif model_type == "multimode_codi":
            return DatasetAdapter._adapt_multimode_codi(
                dataset, model, indices, **model_kwargs
            )
        elif model_type == "multimode_coconut":
            return DatasetAdapter._adapt_multimode_coconut(
                dataset, model, indices, **model_kwargs
            )
        elif model_type == "no_cot":
            return DatasetAdapter._adapt_cot(dataset, model, indices)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    @staticmethod
    def _adapt_cot(dataset, model, indices) -> List[Dict]:
        """Adapt dataset for CoT model."""
        adapted = []

        for idx in indices:
            sample = dataset[idx]

            # CoT just needs the question
            question_text = model.tokenizer.decode(
                sample["question_tokenized"],
                skip_special_tokens=True
            )

            inputs = model.prepare_input(question_text)

            adapted.append({
                **inputs,
                "idx": sample["idx"],
                "question_tokenized": sample["question_tokenized"],
                "steps_tokenized": sample["steps_tokenized"],
                "answer_tokenized": sample["answer_tokenized"]
            })

        return adapted

    @staticmethod
    def _adapt_coconut(
        dataset,
        model,
        indices,
        num_latents: int = 6,
        include_latent_markers: bool = True,
        **kwargs
    ) -> List[Dict]:
        """Adapt dataset for Coconut model.

        Args:
            dataset: Dataset with tokenized samples
            model: Coconut model instance
            indices: Sample indices to adapt
            num_latents: Number of latent tokens (ignored if include_latent_markers=False)
            include_latent_markers: Whether to include <|start-latent|>, <|latent|>, <|end-latent|>.
                If False, formats input like no-CoT for ablation experiments.
        """
        adapted = []

        for idx in indices:
            sample = dataset[idx]

            # Coconut needs question + latent tokens (unless include_latent_markers=False)
            question_text = model.tokenizer.decode(
                sample["question_tokenized"],
                skip_special_tokens=True
            )

            inputs = model.prepare_input(
                question_text,
                num_latents=num_latents,
                include_latent_markers=include_latent_markers
            )

            adapted.append({
                **inputs,
                "idx": sample["idx"],
                "question_tokenized": sample["question_tokenized"],
                "steps_tokenized": sample["steps_tokenized"],
                "answer_tokenized": sample["answer_tokenized"]
            })

        return adapted

    @staticmethod
    def _adapt_codi(
        dataset,
        model,
        indices,
        num_iterations: int = 5,
        **kwargs
    ) -> List[Dict]:
        """Adapt dataset for CODI model."""
        adapted = []

        for idx in indices:
            sample = dataset[idx]

            # CODI needs just the question (latent embeddings are continuous)
            # IMPORTANT: Strip to remove training-format newline added by dataset.py
            question_text = model.tokenizer.decode(
                sample["question_tokenized"],
                skip_special_tokens=True
            ).strip()

            inputs = model.prepare_input(question_text)

            adapted.append({
                **inputs,
                "idx": sample["idx"],
                "num_iterations": num_iterations,
                "question_tokenized": sample["question_tokenized"],
                "steps_tokenized": sample["steps_tokenized"],
                "answer_tokenized": sample["answer_tokenized"]
            })

        return adapted

    @staticmethod
    def _adapt_multimode_codi(
        dataset,
        model,
        indices,
        num_iterations: int = 6,
        mode: str = "latent",
        **kwargs
    ) -> List[Dict]:
        """Adapt dataset for multi-mode CODI model."""
        adapted = []

        for idx in indices:
            sample = dataset[idx]

            # Multimode CODI needs just the question (mode-specific tokens added by generate)
            question_text = model.tokenizer.decode(
                sample["question_tokenized"],
                skip_special_tokens=True
            ).strip()

            inputs = model.prepare_input(question_text, mode=mode)

            adapted.append({
                **inputs,
                "idx": sample["idx"],
                "num_iterations": num_iterations,
                "mode": mode,
                "question_tokenized": sample["question_tokenized"],
                "steps_tokenized": sample["steps_tokenized"],
                "answer_tokenized": sample["answer_tokenized"]
            })

        return adapted

    @staticmethod
    def _adapt_multimode_coconut(
        dataset,
        model,
        indices,
        num_iterations: int = 6,
        mode: str = "latent",
        **kwargs
    ) -> List[Dict]:
        """Adapt dataset for multi-mode Coconut model.

        Similar to multimode_codi but uses the Coconut wrapper without LoRA/projection.
        """
        adapted = []

        for idx in indices:
            sample = dataset[idx]

            # Multimode Coconut needs just the question (mode-specific tokens added by prepare_input)
            question_text = model.tokenizer.decode(
                sample["question_tokenized"],
                skip_special_tokens=True
            ).strip()

            inputs = model.prepare_input(question_text, mode=mode, num_latents=num_iterations)

            adapted.append({
                **inputs,
                "idx": sample["idx"],
                "num_iterations": num_iterations,
                "mode": mode,
                "question_tokenized": sample["question_tokenized"],
                "steps_tokenized": sample["steps_tokenized"],
                "answer_tokenized": sample["answer_tokenized"]
            })

        return adapted


def prepare_sample_for_inference(
    sample: Dict,
    model,
    **model_kwargs
) -> Dict[str, torch.Tensor]:
    """
    Prepare a single sample for inference.

    This is a convenience function for adapting a single sample.

    Args:
        sample: Dataset sample with question_tokenized, steps_tokenized, answer_tokenized
        model: BaseModel instance
        **model_kwargs: Model-specific parameters

    Returns:
        Dictionary with input_ids, attention_mask ready for model.generate()
    """
    # Decode question from tokens
    question_text = model.tokenizer.decode(
        sample["question_tokenized"],
        skip_special_tokens=True
    )

    # Use model's prepare_input
    return model.prepare_input(question_text, **model_kwargs)
