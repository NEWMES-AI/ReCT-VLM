"""
ReCT-VLM: Reasoning-Enhanced CT Vision-Language Model

A comprehensive multi-task learning framework for CT image analysis that combines:
- 3D Vision Encoder with slice-aware and region-aware attention
- Multi-label disease classification with BioBERT
- Text-guided lesion localization
- LLM-based radiology report generation

Repository: https://github.com/NEWMES-AI/ReCT-VLM
"""

__version__ = "0.1.0"
__author__ = "NEWMES-AI"
__license__ = "Apache-2.0"

from .model.vision_encoder import ThreeDVisionEncoder
from .model.classification_head import MultiLabelClassifier
from .model.localization_module import LesionLocalizationModule
from .model.report_generator import ReportGenerator
from .model.multi_task_model import VLM3DMultiTask

__all__ = [
    'ThreeDVisionEncoder',
    'MultiLabelClassifier',
    'LesionLocalizationModule',
    'ReportGenerator',
    'VLM3DMultiTask',
]
