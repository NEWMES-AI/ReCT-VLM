"""
VLM3D Multi-task Learning System

Integrates all components for simultaneous:
1. Multi-label disease classification (18 diseases)
2. Lesion localization (5 key diseases)
3. Clinical report generation

Uses shared vision encoder with task-specific heads.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union

from .vision_encoder import ThreeDVisionEncoder
from .classification_head import MultiLabelClassifier, ClassificationLoss
from .localization_module import LesionLocalizationModule
from .report_generator import ReportGenerator


class VLM3DMultiTask(nn.Module):
    """
    Complete VLM3D multi-task learning system.

    Performs classification, localization, and report generation in one forward pass.
    """

    def __init__(
        self,
        # Vision encoder config
        vision_in_channels: int = 1,
        vision_embed_dim: int = 768,
        vision_depth: int = 12,
        vision_num_heads: int = 12,
        vision_num_regions: int = 20,
        # Classification config
        classification_text_model: str = "emilyalsentzer/Bio_ClinicalBERT",
        num_diseases: int = 18,
        # Localization config
        num_lesion_diseases: int = 5,
        localization_text_model: str = "emilyalsentzer/Bio_ClinicalBERT",
        # Report generation config
        llm_name: str = "meta-llama/Llama-3.1-70B-Instruct",
        num_visual_tokens: int = 256,
        use_lora: bool = True,
        lora_rank: int = 64,
        load_in_4bit: bool = True,
        # Multi-task config
        share_text_encoder: bool = True,
        freeze_vision_encoder: bool = False
    ):
        """
        Args:
            Vision encoder parameters
            Classification parameters
            Localization parameters
            Report generation parameters
            Multi-task parameters
        """
        super().__init__()

        self.vision_embed_dim = vision_embed_dim
        self.num_diseases = num_diseases
        self.num_lesion_diseases = num_lesion_diseases

        # Shared 3D Vision Encoder
        print("Initializing 3D Vision Encoder...")
        self.vision_encoder = ThreeDVisionEncoder(
            in_channels=vision_in_channels,
            embed_dim=vision_embed_dim,
            depth=vision_depth,
            num_heads=vision_num_heads,
            num_regions=vision_num_regions
        )

        if freeze_vision_encoder:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False

        # Task 1: Classification Head
        print("Initializing Classification Head...")
        self.classification_head = MultiLabelClassifier(
            vision_dim=vision_embed_dim,
            text_model_name=classification_text_model,
            num_diseases=num_diseases,
            use_learnable_prompts=True
        )

        # Task 2: Lesion Localization Module
        print("Initializing Localization Module...")
        self.localization_module = LesionLocalizationModule(
            embed_dim=vision_embed_dim,
            num_diseases=num_lesion_diseases,
            text_model_name=localization_text_model if not share_text_encoder else classification_text_model,
            denoising_layers=4,
            unet_base_channels=64
        )

        # Task 3: Report Generator
        print("Initializing Report Generator...")
        self.report_generator = ReportGenerator(
            llm_name=llm_name,
            vision_dim=vision_embed_dim,
            num_visual_tokens=num_visual_tokens,
            use_lora=use_lora,
            lora_rank=lora_rank,
            load_in_4bit=load_in_4bit
        )

        # Task enable flags (for progressive training)
        self.enable_classification = True
        self.enable_localization = True
        self.enable_report_generation = True

    def forward(
        self,
        ct_volume: torch.Tensor,
        segmentation_mask: Optional[torch.Tensor] = None,
        # Classification targets
        disease_labels: Optional[torch.Tensor] = None,
        # Localization targets
        lesion_masks_gt: Optional[Dict[str, torch.Tensor]] = None,
        # Report generation targets
        report_tokens: Optional[torch.Tensor] = None,
        report_prompt: Optional[str] = None,
        # Control flags
        tasks: Optional[List[str]] = None,
        original_shape: Optional[Tuple[int, int, int]] = None
    ) -> Dict[str, Union[torch.Tensor, Dict, List]]:
        """
        Forward pass for multi-task learning.

        Args:
            ct_volume: Input CT volume (B, C, D, H, W)
            segmentation_mask: Anatomical segmentation mask (B, D, H, W)
            disease_labels: Ground truth disease labels (B, num_diseases)
            lesion_masks_gt: Ground truth lesion masks dict
            report_tokens: Tokenized report for training
            report_prompt: Text prompt for report generation
            tasks: Optional list of tasks to perform ["classification", "localization", "generation"]
            original_shape: Original CT volume shape for localization

        Returns:
            Dictionary containing outputs for all tasks
        """
        if tasks is None:
            tasks = []
            if self.enable_classification:
                tasks.append("classification")
            if self.enable_localization:
                tasks.append("localization")
            if self.enable_report_generation:
                tasks.append("generation")

        if original_shape is None:
            original_shape = (ct_volume.size(2), ct_volume.size(3), ct_volume.size(4))

        # Extract vision features (shared across all tasks)
        vision_outputs = self.vision_encoder(ct_volume, segmentation_mask)

        global_features = vision_outputs['global_features']  # (B, embed_dim)
        local_features = vision_outputs['local_features']  # (B, N_patches, embed_dim)
        region_features = vision_outputs['region_features']  # (B, N_regions, embed_dim)

        outputs = {
            'vision_features': vision_outputs
        }

        # Task 1: Classification
        if "classification" in tasks:
            classification_outputs = self.classification_head(
                global_features,
                region_features,
                return_similarities=True
            )
            outputs['classification'] = classification_outputs

        # Task 2: Localization
        if "localization" in tasks:
            localization_outputs = self.localization_module(
                local_features,
                region_features,
                disease_indices=None,  # All diseases
                original_shape=original_shape
            )
            outputs['localization'] = localization_outputs

        # Task 3: Report Generation
        if "generation" in tasks:
            # Use classification and localization results if available
            class_predictions = outputs.get('classification', {}).get('probabilities', None)
            lesion_masks_pred = outputs.get('localization', None)

            report_outputs = self.report_generator(
                global_features=global_features,
                local_features=local_features,
                region_features=region_features,
                lesion_masks=lesion_masks_pred if lesion_masks_pred else lesion_masks_gt,
                class_predictions=class_predictions,
                prompt=report_prompt,
                labels=report_tokens
            )
            outputs['generation'] = report_outputs

        return outputs

    @torch.no_grad()
    def predict(
        self,
        ct_volume: torch.Tensor,
        segmentation_mask: Optional[torch.Tensor] = None,
        classification_threshold: float = 0.5,
        localization_threshold: float = 0.5,
        generate_report: bool = True,
        report_generation_config: Optional[Dict] = None
    ) -> Dict[str, Union[torch.Tensor, List, Dict]]:
        """
        Inference mode: generate all predictions.

        Args:
            ct_volume: Input CT volume (B, C, D, H, W)
            segmentation_mask: Optional anatomical mask
            classification_threshold: Threshold for disease classification
            localization_threshold: Threshold for lesion segmentation
            generate_report: Whether to generate clinical report
            report_generation_config: Optional config for report generation

        Returns:
            Complete predictions for all tasks
        """
        original_shape = (ct_volume.size(2), ct_volume.size(3), ct_volume.size(4))

        # Forward pass
        tasks = ["classification", "localization"]
        if generate_report:
            tasks.append("generation")

        outputs = self.forward(
            ct_volume=ct_volume,
            segmentation_mask=segmentation_mask,
            tasks=tasks,
            original_shape=original_shape
        )

        # Process classification predictions
        class_outputs = outputs['classification']
        class_probs = class_outputs['probabilities']
        class_preds = (class_probs > classification_threshold).long()

        predicted_diseases = []
        for batch_preds in class_preds:
            batch_diseases = [
                self.classification_head.DISEASE_NAMES[i]
                for i, pred in enumerate(batch_preds)
                if pred == 1
            ]
            predicted_diseases.append(batch_diseases)

        # Process localization predictions
        lesion_masks = outputs['localization']
        binary_masks = {
            disease: (mask > localization_threshold).float()
            for disease, mask in lesion_masks.items()
        }

        # Generate report if requested
        reports = None
        if generate_report:
            if report_generation_config is None:
                report_generation_config = {
                    'max_new_tokens': 512,
                    'temperature': 0.7,
                    'top_p': 0.9,
                    'do_sample': True
                }

            # Extract features again (we need them for generation)
            vision_outputs = outputs['vision_features']

            reports = self.report_generator.generate(
                global_features=vision_outputs['global_features'],
                local_features=vision_outputs['local_features'],
                region_features=vision_outputs['region_features'],
                lesion_masks=lesion_masks,
                class_predictions=class_probs,
                **report_generation_config
            )

        return {
            'classification': {
                'probabilities': class_probs,
                'predictions': class_preds,
                'disease_names': predicted_diseases
            },
            'localization': {
                'masks': lesion_masks,
                'binary_masks': binary_masks
            },
            'reports': reports
        }

    def freeze_task(self, task: str):
        """Freeze parameters for a specific task."""
        if task == "classification":
            for param in self.classification_head.parameters():
                param.requires_grad = False
        elif task == "localization":
            for param in self.localization_module.parameters():
                param.requires_grad = False
        elif task == "generation":
            for param in self.report_generator.parameters():
                param.requires_grad = False
        elif task == "vision":
            for param in self.vision_encoder.parameters():
                param.requires_grad = False

    def unfreeze_task(self, task: str):
        """Unfreeze parameters for a specific task."""
        if task == "classification":
            for param in self.classification_head.parameters():
                param.requires_grad = True
        elif task == "localization":
            for param in self.localization_module.parameters():
                param.requires_grad = True
        elif task == "generation":
            for param in self.report_generator.parameters():
                param.requires_grad = True
        elif task == "vision":
            for param in self.vision_encoder.parameters():
                param.requires_grad = True

    def get_trainable_parameters(self) -> Dict[str, int]:
        """Get count of trainable parameters per module."""
        def count_params(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        return {
            'vision_encoder': count_params(self.vision_encoder),
            'classification_head': count_params(self.classification_head),
            'localization_module': count_params(self.localization_module),
            'report_generator': count_params(self.report_generator),
            'total': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss function with task weighting.
    """

    def __init__(
        self,
        # Task weights
        weight_classification: float = 1.0,
        weight_localization: float = 2.0,
        weight_generation: float = 1.5,
        # Classification config
        classification_pos_weight: Optional[torch.Tensor] = None,
        use_focal_loss: bool = False,
        # Localization config
        dice_weight: float = 1.0,
        bce_weight: float = 1.0,
        # Dynamic weighting
        use_gradnorm: bool = False
    ):
        """
        Args:
            weight_classification: Weight for classification loss
            weight_localization: Weight for localization loss
            weight_generation: Weight for report generation loss
            classification_pos_weight: Positive class weights for classification
            use_focal_loss: Use focal loss for classification
            dice_weight: Weight for Dice loss in localization
            bce_weight: Weight for BCE loss in localization
            use_gradnorm: Use GradNorm for dynamic task weighting
        """
        super().__init__()

        # Task weights (learnable if using GradNorm)
        if use_gradnorm:
            self.weight_classification = nn.Parameter(torch.tensor(weight_classification))
            self.weight_localization = nn.Parameter(torch.tensor(weight_localization))
            self.weight_generation = nn.Parameter(torch.tensor(weight_generation))
        else:
            self.register_buffer('weight_classification', torch.tensor(weight_classification))
            self.register_buffer('weight_localization', torch.tensor(weight_localization))
            self.register_buffer('weight_generation', torch.tensor(weight_generation))

        self.use_gradnorm = use_gradnorm

        # Classification loss
        from .classification_head import ClassificationLoss
        self.classification_loss_fn = ClassificationLoss(
            pos_weight=classification_pos_weight,
            use_focal_loss=use_focal_loss
        )

        # Localization loss weights
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def dice_loss(self, pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-5) -> torch.Tensor:
        """Dice loss for segmentation."""
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)

        intersection = (pred * target).sum()
        dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

        return 1 - dice

    def forward(
        self,
        outputs: Dict,
        targets: Dict,
        tasks: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss.

        Args:
            outputs: Model outputs dictionary
            targets: Ground truth targets dictionary
            tasks: Optional list of tasks to compute loss for

        Returns:
            Dictionary with total loss and per-task losses
        """
        losses = {}
        total_loss = 0.0

        if tasks is None:
            tasks = []
            if 'classification' in outputs:
                tasks.append('classification')
            if 'localization' in outputs:
                tasks.append('localization')
            if 'generation' in outputs:
                tasks.append('generation')

        # Task 1: Classification Loss
        if 'classification' in tasks and 'classification' in outputs:
            class_outputs = outputs['classification']
            class_targets = targets.get('disease_labels')

            if class_targets is not None:
                loss_cls = self.classification_loss_fn(
                    class_outputs['logits'],
                    class_targets
                )
                losses['classification'] = loss_cls
                total_loss = total_loss + self.weight_classification * loss_cls

        # Task 2: Localization Loss
        if 'localization' in tasks and 'localization' in outputs:
            loc_outputs = outputs['localization']
            loc_targets = targets.get('lesion_masks')

            if loc_targets is not None:
                loss_loc = 0.0
                num_diseases = 0

                for disease, pred_mask in loc_outputs.items():
                    if disease in loc_targets:
                        gt_mask = loc_targets[disease]

                        # Dice loss
                        loss_dice = self.dice_loss(
                            torch.sigmoid(pred_mask),
                            gt_mask.float()
                        )

                        # BCE loss
                        loss_bce = F.binary_cross_entropy_with_logits(
                            pred_mask,
                            gt_mask.float()
                        )

                        loss_loc = loss_loc + (self.dice_weight * loss_dice + self.bce_weight * loss_bce)
                        num_diseases += 1

                if num_diseases > 0:
                    loss_loc = loss_loc / num_diseases
                    losses['localization'] = loss_loc
                    total_loss = total_loss + self.weight_localization * loss_loc

        # Task 3: Report Generation Loss
        if 'generation' in tasks and 'generation' in outputs:
            gen_outputs = outputs['generation']

            if 'loss' in gen_outputs:
                loss_gen = gen_outputs['loss']
                losses['generation'] = loss_gen
                total_loss = total_loss + self.weight_generation * loss_gen

        losses['total'] = total_loss

        return losses


# Test code
if __name__ == "__main__":
    print("Testing VLM3D Multi-task Model...")

    # Small test configuration
    config = {
        'vision_in_channels': 1,
        'vision_embed_dim': 256,  # Smaller for testing
        'vision_depth': 2,  # Fewer layers
        'vision_num_heads': 8,
        'vision_num_regions': 20,
        'num_diseases': 18,
        'num_lesion_diseases': 5,
        'classification_text_model': 'distilbert-base-uncased',
        'localization_text_model': 'distilbert-base-uncased',
        'llm_name': 'gpt2',  # Small model for testing
        'num_visual_tokens': 32,  # Fewer tokens
        'use_lora': False,  # Disable for testing
        'load_in_4bit': False
    }

    print("\n1. Creating model...")
    # Note: This will take time as it loads the LLM
    # model = VLM3DMultiTask(**config)

    print("   Skipping full model creation (requires LLM)")
    print("   Testing individual components instead...")

    # Test multi-task loss
    print("\n2. Testing MultiTaskLoss...")

    loss_fn = MultiTaskLoss(
        weight_classification=1.0,
        weight_localization=2.0,
        weight_generation=1.5
    )

    # Create dummy outputs and targets
    batch_size = 2
    outputs = {
        'classification': {
            'logits': torch.randn(batch_size, 18)
        },
        'localization': {
            'pericardial_effusion': torch.randn(batch_size, 64, 512, 512),
            'pleural_effusion': torch.randn(batch_size, 64, 512, 512)
        }
    }

    targets = {
        'disease_labels': torch.randint(0, 2, (batch_size, 18)).float(),
        'lesion_masks': {
            'pericardial_effusion': torch.randint(0, 2, (batch_size, 64, 512, 512)).float(),
            'pleural_effusion': torch.randint(0, 2, (batch_size, 64, 512, 512)).float()
        }
    }

    losses = loss_fn(outputs, targets, tasks=['classification', 'localization'])

    print(f"   Total loss: {losses['total'].item():.4f}")
    print(f"   Classification loss: {losses.get('classification', torch.tensor(0.0)).item():.4f}")
    print(f"   Localization loss: {losses.get('localization', torch.tensor(0.0)).item():.4f}")

    print("\n✓ Tests passed!")
    print("\nNote: Full model testing requires LLM and significant GPU memory")
    print("Use with actual model for complete testing")
