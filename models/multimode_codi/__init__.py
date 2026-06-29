"""
Multi-mode CODI module.

Supports three inference modes:
- Direct: prompt + eot_id + eocot_id → answer (no reasoning)
- Verbalized: prompt + eot_id + bocot_id → [CoT tokens] + eocot_id + answer
- Latent: prompt + eot_id + bocot_id → [N latent iterations] → eocot_id → answer
"""

from models.multimode_codi.multimode_codi_wrapper import MultimodeCODIInference
from models.multimode_codi.multimode_codi_model import MultimodeCODIModel

__all__ = ['MultimodeCODIInference', 'MultimodeCODIModel']
