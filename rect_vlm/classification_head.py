"""
Multi-label Disease Classification Head

Text-prompt based multi-label classifier for 18 major diseases.
Uses cosine similarity between vision features and disease text embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from transformers import AutoTokenizer, AutoModel


class MultiLabelClassifier(nn.Module):
    """
    Text-prompt based multi-label disease classifier.

    Uses pre-trained medical language model (BioBERT/ClinicalBERT) to encode
    disease prompts, then computes cosine similarity with vision features.

    Args:
        vision_dim: Dimension of vision features (default: 768)
        text_model_name: HuggingFace model name for text encoder
        num_diseases: Number of diseases to classify (default: 18)
        temperature: Temperature scaling parameter for similarity
        use_learnable_prompts: Whether to use learnable soft prompts
    """

    # 18 target diseases
    DISEASE_NAMES = [
        "lung_nodule",
        "pleural_effusion",
        "consolidation",
        "ground_glass_opacity",
        "pericardial_effusion",
        "pneumothorax",
        "atelectasis",
        "pneumonia",
        "pulmonary_edema",
        "emphysema",
        "fibrosis",
        "mass",
        "cardiomegaly",
        "mediastinal_widening",
        "pleural_thickening",
        "fracture",
        "calcification",
        "lymphadenopathy"
    ]

    # Prompt templates for each disease
    PROMPT_TEMPLATES = {
        "lung_nodule": "A CT scan showing a lung nodule, which is a round or oval-shaped growth in the lung tissue",
        "pleural_effusion": "Evidence of pleural effusion, which is fluid accumulation in the pleural space between the lung and chest wall",
        "consolidation": "CT findings of consolidation, indicating alveolar airspaces filled with fluid, pus, blood, or cells",
        "ground_glass_opacity": "Ground-glass opacity visible on CT, showing hazy increased lung attenuation with preserved bronchial and vascular markings",
        "pericardial_effusion": "Pericardial effusion present, which is fluid accumulation in the pericardial cavity surrounding the heart",
        "pneumothorax": "Pneumothorax detected on CT scan, showing air in the pleural space causing partial or complete lung collapse",
        "atelectasis": "Evidence of atelectasis, which is collapse or incomplete expansion of lung tissue",
        "pneumonia": "CT findings consistent with pneumonia, showing inflammatory consolidation of lung parenchyma",
        "pulmonary_edema": "Pulmonary edema visible on CT, indicating fluid accumulation in the lung interstitium and alveoli",
        "emphysema": "Emphysema present, showing destruction of alveolar walls and permanent enlargement of airspaces",
        "fibrosis": "Pulmonary fibrosis detected, with thickening and scarring of lung tissue",
        "mass": "A mass lesion visible on CT scan, which is a large solid lesion greater than 3 cm in diameter",
        "cardiomegaly": "Cardiomegaly present, indicating enlargement of the heart visible on CT imaging",
        "mediastinal_widening": "Mediastinal widening observed, showing enlargement of the mediastinal space",
        "pleural_thickening": "Pleural thickening detected, indicating increased thickness of the pleural membranes",
        "fracture": "Evidence of rib or vertebral fracture visible on CT scan",
        "calcification": "Calcification present, showing areas of calcium deposits in lung tissue or vessels",
        "lymphadenopathy": "Lymphadenopathy detected, indicating enlarged lymph nodes in the thorax"
    }

    def __init__(
        self,
        vision_dim: int = 768,
        text_model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
        num_diseases: int = 18,
        temperature: float = 0.07,
        use_learnable_prompts: bool = True,
        freeze_text_encoder: bool = True
    ):
        super().__init__()

        self.vision_dim = vision_dim
        self.num_diseases = num_diseases
        self.use_learnable_prompts = use_learnable_prompts

        # Text encoder (BioBERT or ClinicalBERT)
        print(f"Loading text encoder: {text_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.text_encoder = AutoModel.from_pretrained(text_model_name)

        if freeze_text_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False

        # Get text encoder output dimension
        text_dim = self.text_encoder.config.hidden_size

        # Projection layers to align vision and text features
        self.vision_proj = nn.Sequential(
            nn.Linear(vision_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512)
        )

        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512)
        )

        # Learnable temperature parameter
        self.temperature = nn.Parameter(torch.tensor(temperature))

        # Learnable soft prompts (optional)
        if use_learnable_prompts:
            self.soft_prompts = nn.Parameter(
                torch.randn(num_diseases, 512)
            )
            nn.init.normal_(self.soft_prompts, std=0.02)

        # Disease-specific bias terms
        self.disease_bias = nn.Parameter(torch.zeros(num_diseases))

        # Pre-compute and cache disease text embeddings
        self.register_buffer('disease_text_embeds', torch.zeros(num_diseases, 512))
        self._initialize_disease_embeddings()

    def _initialize_disease_embeddings(self):
        """Pre-compute text embeddings for all disease prompts."""
        print("Initializing disease text embeddings...")

        with torch.no_grad():
            for idx, disease in enumerate(self.DISEASE_NAMES[:self.num_diseases]):
                prompt = self.PROMPT_TEMPLATES[disease]
                text_embed = self._encode_text(prompt)
                self.disease_text_embeds[idx] = text_embed

        print(f"Initialized {self.num_diseases} disease embeddings")

    def _encode_text(self, text: str) -> torch.Tensor:
        """Encode text using the text encoder."""
        tokens = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(self.text_encoder.device)

        outputs = self.text_encoder(**tokens)
        # Use [CLS] token embedding
        text_features = outputs.last_hidden_state[:, 0, :]  # (1, text_dim)

        # Project to common space
        text_features = self.text_proj(text_features)  # (1, 512)

        # L2 normalize
        text_features = F.normalize(text_features, p=2, dim=-1)

        return text_features.squeeze(0)  # (512,)

    def forward(
        self,
        global_features: torch.Tensor,
        region_features: Optional[torch.Tensor] = None,
        return_similarities: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for disease classification.

        Args:
            global_features: Global vision features (B, vision_dim)
            region_features: Optional region features (B, N_regions, vision_dim)
            return_similarities: Whether to return raw similarity scores

        Returns:
            Dictionary containing:
                - logits: Classification logits (B, num_diseases)
                - probabilities: Sigmoid probabilities (B, num_diseases)
                - similarities: Optional raw cosine similarities
        """
        B = global_features.size(0)

        # Option 1: Use only global features
        if region_features is None:
            vision_features = global_features  # (B, vision_dim)
        else:
            # Option 2: Combine global and region features
            # Average pool region features
            region_pooled = region_features.mean(dim=1)  # (B, vision_dim)
            vision_features = (global_features + region_pooled) / 2

        # Project vision features to common space
        vision_features = self.vision_proj(vision_features)  # (B, 512)

        # L2 normalize
        vision_features = F.normalize(vision_features, p=2, dim=-1)  # (B, 512)

        # Get disease text embeddings
        text_features = self.disease_text_embeds  # (num_diseases, 512)

        # Add learnable soft prompts if enabled
        if self.use_learnable_prompts:
            text_features = text_features + self.soft_prompts
            text_features = F.normalize(text_features, p=2, dim=-1)

        # Compute cosine similarity
        # (B, 512) @ (512, num_diseases) -> (B, num_diseases)
        similarities = vision_features @ text_features.T

        # Scale by temperature and add bias
        logits = similarities / self.temperature.clamp(min=0.01) + self.disease_bias

        # Apply sigmoid for multi-label classification
        probabilities = torch.sigmoid(logits)

        outputs = {
            'logits': logits,
            'probabilities': probabilities
        }

        if return_similarities:
            outputs['similarities'] = similarities

        return outputs

    def predict(
        self,
        global_features: torch.Tensor,
        region_features: Optional[torch.Tensor] = None,
        threshold: float = 0.5
    ) -> Dict[str, torch.Tensor]:
        """
        Make predictions with optional thresholding.

        Args:
            global_features: Global vision features (B, vision_dim)
            region_features: Optional region features
            threshold: Classification threshold (default: 0.5)

        Returns:
            Dictionary containing predictions and probabilities
        """
        with torch.no_grad():
            outputs = self.forward(global_features, region_features)
            predictions = (outputs['probabilities'] > threshold).long()

            return {
                'predictions': predictions,
                'probabilities': outputs['probabilities'],
                'disease_names': [
                    [self.DISEASE_NAMES[i] for i, pred in enumerate(batch) if pred == 1]
                    for batch in predictions
                ]
            }

    def get_class_weights(self, label_counts: torch.Tensor) -> torch.Tensor:
        """
        Compute class weights for handling class imbalance.

        Args:
            label_counts: Number of positive samples per class (num_diseases,)

        Returns:
            Class weights for loss function (num_diseases,)
        """
        total_samples = label_counts.sum()
        class_weights = total_samples / (self.num_diseases * label_counts.clamp(min=1))
        return class_weights


class ClassificationLoss(nn.Module):
    """
    Loss function for multi-label classification with class weighting.
    """

    def __init__(
        self,
        pos_weight: Optional[torch.Tensor] = None,
        use_focal_loss: bool = False,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0
    ):
        super().__init__()

        self.pos_weight = pos_weight
        self.use_focal_loss = use_focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute classification loss.

        Args:
            logits: Predicted logits (B, num_diseases)
            targets: Ground truth labels (B, num_diseases)

        Returns:
            Loss value
        """
        if self.use_focal_loss:
            return self._focal_loss(logits, targets)
        else:
            return F.binary_cross_entropy_with_logits(
                logits,
                targets,
                pos_weight=self.pos_weight
            )

    def _focal_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Focal loss for handling class imbalance.
        """
        probs = torch.sigmoid(logits)
        bce_loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction='none'
        )

        # Compute focal weights
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.focal_gamma

        # Apply alpha weighting
        alpha_weight = torch.where(
            targets == 1,
            self.focal_alpha,
            1 - self.focal_alpha
        )

        loss = alpha_weight * focal_weight * bce_loss
        return loss.mean()


# Test code
if __name__ == "__main__":
    print("Testing MultiLabelClassifier...")

    # Create dummy data
    batch_size = 2
    vision_dim = 768
    num_regions = 20
    num_diseases = 18

    global_features = torch.randn(batch_size, vision_dim)
    region_features = torch.randn(batch_size, num_regions, vision_dim)

    # Test classifier (use distilbert for faster testing)
    classifier = MultiLabelClassifier(
        vision_dim=vision_dim,
        text_model_name="distilbert-base-uncased",  # Faster for testing
        num_diseases=num_diseases,
        use_learnable_prompts=True
    )

    print("\nForward pass test:")
    outputs = classifier(global_features, region_features, return_similarities=True)

    print(f"  Logits shape: {outputs['logits'].shape}")
    print(f"  Probabilities shape: {outputs['probabilities'].shape}")
    print(f"  Similarities shape: {outputs['similarities'].shape}")

    print(f"\n  Sample probabilities: {outputs['probabilities'][0][:5]}")

    # Test prediction
    print("\nPrediction test:")
    predictions = classifier.predict(global_features, region_features, threshold=0.5)

    print(f"  Predictions shape: {predictions['predictions'].shape}")
    print(f"  First sample predictions: {predictions['predictions'][0]}")
    print(f"  First sample diseases: {predictions['disease_names'][0]}")

    # Test loss
    print("\nLoss function test:")
    targets = torch.randint(0, 2, (batch_size, num_diseases)).float()

    loss_fn = ClassificationLoss()
    loss = loss_fn(outputs['logits'], targets)

    print(f"  Loss: {loss.item():.4f}")

    # Test focal loss
    focal_loss_fn = ClassificationLoss(use_focal_loss=True)
    focal_loss = focal_loss_fn(outputs['logits'], targets)

    print(f"  Focal loss: {focal_loss.item():.4f}")

    print("\n✓ All tests passed!")
