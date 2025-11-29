# Sub-objective 3: Multi-task Learning System Architecture

## Overview

This document describes the architecture for Sub-objective 3, which extends the 3D Vision Encoder with three main capabilities:
1. **Multi-label Disease Classification**: 18 major disease classification
2. **Lesion Localization**: Precise localization for 5 key diseases
3. **Report Generation**: Automatic clinical report generation using LLMs

The system performs all three tasks simultaneously using a unified multi-task learning framework.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Input: 3D CT Volume (D×H×W)                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                  3D Vision Encoder (from Phase 3.1)              │
│  • Global features: (B, 768)                                     │
│  • Local features: (B, N_patches, 768)                           │
│  • Region features: (B, N_regions, 768)                          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
         ┌────────────────────┼────────────────────┐
         ↓                    ↓                    ↓
┌────────────────┐  ┌────────────────────┐  ┌─────────────────┐
│ Classification │  │ Lesion Localization│  │ Report Generator│
│     Head       │  │   Module (3-stage) │  │   (VLM/LLM)     │
└────────────────┘  └────────────────────┘  └─────────────────┘
         ↓                    ↓                    ↓
┌────────────────┐  ┌────────────────────┐  ┌─────────────────┐
│  18 Disease    │  │  Segmentation Maps │  │ Clinical Report │
│  Predictions   │  │  (5 key diseases)  │  │   (Text)        │
└────────────────┘  └────────────────────┘  └─────────────────┘
```

## Module 1: Multi-label Disease Classification

### Objective
Classify presence of 18 major diseases from CT volumes using text-prompt based multi-label classification.

### 18 Target Diseases
1. Lung nodule
2. Pleural effusion
3. Consolidation
4. Ground-glass opacity
5. Pericardial effusion
6. Pneumothorax
7. Atelectasis
8. Pneumonia
9. Pulmonary edema
10. Emphysema
11. Fibrosis
12. Mass
13. Cardiomegaly
14. Mediastinal widening
15. Pleural thickening
16. Fracture
17. Calcification
18. Lymphadenopathy

### Architecture

```python
# Text prompt embeddings (learned)
disease_prompts = [
    "A CT scan showing {disease_name}",
    "Evidence of {disease_name} in chest CT",
    ...
]

# For each disease:
text_embed = TextEncoder(prompt)  # (768,)
vision_global = VisionEncoder.global_features  # (B, 768)

# Cosine similarity classification
similarity = cosine_sim(vision_global, text_embed)  # (B,)
prob = sigmoid(similarity)  # Multi-label
```

### Implementation Details
- **Text Encoder**: BioBERT or ClinicalBERT for medical domain
- **Classification Method**: Cosine similarity + temperature scaling
- **Loss Function**: Binary Cross-Entropy with class weights
- **Prompt Learning**: Learnable soft prompts per disease

### Module Components
```
ClassificationHead
├── TextEncoder (BioBERT-base)
│   └── Prompt templates (18 diseases)
├── Similarity Layer
│   ├── Cosine similarity computation
│   └── Temperature parameter (learnable)
└── Output Layer
    └── Multi-label sigmoid (18 classes)
```

## Module 2: Lesion Localization

### Objective
Generate precise segmentation masks for 5 key diseases using a 3-stage approach with weak supervision from text descriptions.

### 5 Target Diseases
1. Pericardial effusion
2. Pleural effusion
3. Consolidation
4. Ground-glass opacity
5. Lung nodules

### 3-Stage Architecture

#### Stage 1: Text Representation Redefinition
- **Purpose**: Convert disease text to rich semantic embeddings
- **Method**:
  - Use disease-specific prompts with anatomical context
  - Example: "Pericardial effusion: fluid accumulation in the pericardial cavity, visible as low-density region surrounding the heart"
  - Text encoder: Clinical BERT or domain-specific language model
  - Output: Disease text embeddings (D_text, 768)

```python
text_prompts = {
    "pericardial_effusion": "Pericardial effusion: fluid accumulation...",
    "pleural_effusion": "Pleural effusion: fluid in pleural space...",
    ...
}

text_features = text_encoder(prompts)  # (5, 768)
```

#### Stage 2: 3D CT Representation Compression with Denoising
- **Purpose**: Compress local features while preserving lesion information
- **Method**:
  - Input: Local patch features (B, N_patches, 768)
  - Denoising transformer blocks to remove noise
  - Spatial compression to manageable resolution
  - Output: Compressed features (B, 768, D', H', W')

```python
# Denoising Transformer
denoised_features = DenoisingTransformer(local_features)

# Reshape to 3D grid
feature_3d = reshape_to_3d(denoised_features)  # (B, 768, D', H', W')

# Optional: U-Net encoder path for further compression
compressed = UNetEncoder(feature_3d)  # (B, 768, D'', H'', W'')
```

**Denoising Strategy**:
- Self-attention across patches to enhance lesion signals
- Cross-attention to disease text features for guidance
- Residual connections to preserve fine details

#### Stage 3: Attention U-Net Based Activation Map Generation
- **Purpose**: Generate high-resolution segmentation masks
- **Architecture**: Attention U-Net with text-guided attention
- **Method**:
  - Encoder: Standard U-Net encoder path
  - Decoder: U-Net decoder with attention gates
  - Text-guided attention: Use disease text features to modulate attention
  - Multi-scale fusion for precise boundaries

```python
# Text-guided Attention U-Net
class TextGuidedAttentionUNet:
    def __init__(self):
        self.encoder = UNetEncoder()  # Conv blocks with downsampling
        self.decoder = UNetDecoder()  # Conv blocks with upsampling
        self.attention_gates = AttentionGates()  # Text-modulated
        self.text_injector = CrossAttention()

    def forward(self, features, text_embeds, disease_idx):
        # Encoder path
        enc1, enc2, enc3, enc4, bottleneck = self.encoder(features)

        # Inject text guidance at multiple scales
        bottleneck = self.text_injector(bottleneck, text_embeds[disease_idx])

        # Decoder path with attention gates
        dec4 = self.decoder.up1(bottleneck)
        dec4 = self.attention_gates.gate4(dec4, enc4, text_embeds[disease_idx])
        dec4 = torch.cat([dec4, enc4], dim=1)

        dec3 = self.decoder.up2(dec4)
        dec3 = self.attention_gates.gate3(dec3, enc3, text_embeds[disease_idx])
        dec3 = torch.cat([dec3, enc3], dim=1)

        # ... continue for dec2, dec1

        # Final segmentation
        seg_map = self.final_conv(dec1)  # (B, 1, D, H, W)
        return seg_map
```

**Attention Gate Design**:
```python
class TextGuidedAttentionGate:
    def forward(self, decoder_features, encoder_features, text_features):
        # Combine features
        combined = W_d(decoder_features) + W_e(encoder_features) + W_t(text_features)
        attention_weights = sigmoid(W_attn(relu(combined)))

        # Apply attention
        gated_features = encoder_features * attention_weights
        return gated_features
```

### Text-to-Localization Framework
- **Weak Supervision**: Use text descriptions to guide localization without pixel-level annotations
- **Cross-Modal Alignment**: Align visual features with disease text embeddings
- **Multi-Scale Supervision**: Apply losses at multiple resolution levels

### Loss Functions
1. **Segmentation Loss**: Dice Loss + Binary Cross-Entropy
2. **Text-Vision Alignment Loss**: Contrastive loss between predicted mask features and text embeddings
3. **Consistency Loss**: Ensure consistency across scales

```python
# Total localization loss
loss_seg = dice_loss(pred_mask, gt_mask) + bce_loss(pred_mask, gt_mask)
loss_align = contrastive_loss(mask_features, text_features)
loss_consistency = mse_loss(low_res_pred, downsample(high_res_pred))

loss_localization = loss_seg + 0.1 * loss_align + 0.05 * loss_consistency
```

## Module 3: Report Generation

### Objective
Generate clinical radiology reports from CT volumes using Large Language Models (LLMs) or Vision-Language Models (VLMs).

### Model Selection Options

#### Option 1: LLM-based (Llama/Gemma)
- **Base Model**: Llama-3.1-70B or Gemma-2-27B
- **Approach**: CT features → MLP Projector → LLM input space
- **Advantages**: High text quality, medical reasoning capability
- **Challenges**: Requires vision-language alignment training

#### Option 2: VLM-based (Recommended)
- **Base Model**: LLaVA-Med, Med-Flamingo, or BiomedCLIP-based VLM
- **Approach**: Direct image-text generation
- **Advantages**: Pre-trained vision-language alignment
- **Challenges**: May need domain adaptation for CT

### Architecture (VLM-based Approach)

```
┌─────────────────────────────────────────────────────────────┐
│                   3D Vision Encoder                          │
│  • Global features: (B, 768)                                │
│  • Local features: (B, N_patches, 768)                       │
│  • Region features: (B, N_regions, 768)                      │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                   Vision Projector                           │
│  • MLP: 768 → 4096 → 4096                                   │
│  • Projects to LLM embedding space                           │
│  • Output: (B, N_visual_tokens, 4096)                       │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│            Optional: Lesion Location Injector                │
│  • Inject lesion location text as soft prompts              │
│  • Example: "[FINDING] Pleural effusion in right lower lobe"│
│  • Use classification + localization results                │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                   Language Model (LLM/VLM)                   │
│  • Llama-3.1-70B-Instruct or Gemma-2-27B-it                 │
│  • Input: [visual_tokens] + [text_prompt]                   │
│  • Output: Clinical report text                             │
│  • LoRA fine-tuning on CT-RATE reports                      │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                   Post-processing                            │
│  • Format as structured report (Findings, Impression)        │
│  • Inject specific lesion locations from localization module │
└─────────────────────────────────────────────────────────────┘
```

### MLP Projector Design

```python
class VisionToLLMProjector(nn.Module):
    def __init__(self, vision_dim=768, llm_dim=4096, num_visual_tokens=256):
        super().__init__()

        # Multi-layer projector
        self.projector = nn.Sequential(
            nn.Linear(vision_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
            nn.LayerNorm(llm_dim)
        )

        # Learnable visual token queries (like Q-Former in BLIP-2)
        self.visual_queries = nn.Parameter(
            torch.randn(1, num_visual_tokens, vision_dim)
        )

        # Cross-attention to compress patch features
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=vision_dim,
            num_heads=12,
            batch_first=True
        )

    def forward(self, global_features, local_features, region_features):
        # Option 1: Simple projection of global features
        # visual_tokens = self.projector(global_features).unsqueeze(1)

        # Option 2: Query-based compression (better)
        B = local_features.size(0)
        queries = self.visual_queries.expand(B, -1, -1)

        # Cross-attention: queries attend to local + region features
        all_features = torch.cat([local_features, region_features], dim=1)
        visual_embeds, _ = self.cross_attn(
            query=queries,
            key=all_features,
            value=all_features
        )

        # Project to LLM space
        visual_tokens = self.projector(visual_embeds)  # (B, 256, 4096)

        return visual_tokens
```

### Lesion Location Injection

Two strategies to inject lesion location information:

#### Strategy 1: Text-based Injection (Recommended)
```python
# Generate location text from localization module
location_texts = []
for disease_idx in detected_diseases:
    mask = localization_module.get_mask(disease_idx)
    location = extract_location_description(mask)
    location_texts.append(f"[FINDING] {disease_names[disease_idx]} detected in {location}")

# Prepend to prompt
prompt = "\n".join(location_texts) + "\n\nGenerate a clinical radiology report:"
```

#### Strategy 2: Cross-Modal Attention
```python
# Use lesion masks as attention bias
lesion_features = extract_features_from_masks(lesion_masks)  # (B, 5, 768)
lesion_embeds = lesion_projector(lesion_features)  # (B, 5, 4096)

# Inject via cross-attention in LLM
# Modify LLM attention to attend to lesion embeddings
```

### Fine-tuning Strategy

#### Stage 1: Vision-Language Alignment (2-5 epochs)
- **Freeze**: LLM parameters
- **Train**: Vision projector only
- **Loss**: Next-token prediction on report text
- **Data**: CT-RATE reports + OmniAbnorm CoT data

#### Stage 2: LoRA Fine-tuning (10-20 epochs)
- **Freeze**: Vision encoder backbone
- **Train**: Vision projector + LoRA adapters in LLM
- **LoRA Config**: rank=64, alpha=128, target modules=[q_proj, v_proj, o_proj]
- **Loss**: Next-token prediction + reinforcement learning

#### Stage 3: Reinforcement Learning (Optional, 5-10 epochs)
- **Method**: GRPO (Group Relative Policy Optimization) or DAPO
- **Reward Model**:
  - Clinical accuracy (via reference report similarity)
  - Factual consistency (via fact extraction + verification)
  - Fluency (language model perplexity)
  - Format adherence (structured report format)

```python
# GRPO pseudocode
for batch in dataloader:
    # Generate multiple reports (K=4)
    reports = [model.generate() for _ in range(K)]

    # Compute rewards
    rewards = [reward_model(report, reference) for report in reports]

    # Compute advantages relative to group mean
    advantages = [r - mean(rewards) for r in rewards]

    # Policy gradient update
    loss = -sum(log_prob(report) * advantage for report, advantage in zip(reports, advantages))
    loss.backward()
```

### Report Format

Standard structured clinical report:
```
FINDINGS:
- <Finding 1 with location>
- <Finding 2 with location>
- ...

IMPRESSION:
<Summary and clinical significance>

RECOMMENDATIONS:
<Follow-up recommendations if any>
```

## Module 4: Multi-task Learning Integration

### Unified Training Framework

```python
class VLM3DMultiTask(nn.Module):
    def __init__(self):
        # Shared backbone
        self.vision_encoder = ThreeDVisionEncoder()

        # Task-specific heads
        self.classification_head = MultiLabelClassifier(num_classes=18)
        self.localization_module = LesionLocalizationModule(num_diseases=5)
        self.report_generator = ReportGenerator(llm_name="llama-3.1-70b")

    def forward(self, ct_volume, seg_mask=None):
        # Shared feature extraction
        features = self.vision_encoder(ct_volume, seg_mask)

        # Task 1: Classification
        class_logits = self.classification_head(
            features['global_features'],
            features['region_features']
        )

        # Task 2: Localization
        lesion_masks = self.localization_module(
            features['local_features'],
            features['region_features']
        )

        # Task 3: Report Generation
        report = self.report_generator(
            features=features,
            class_predictions=class_logits,
            lesion_info=lesion_masks
        )

        return {
            'classification': class_logits,
            'localization': lesion_masks,
            'report': report
        }
```

### Multi-task Loss Function

```python
def compute_multi_task_loss(outputs, targets, phase='training'):
    # Task 1: Classification Loss
    loss_cls = F.binary_cross_entropy_with_logits(
        outputs['classification'],
        targets['class_labels'],
        pos_weight=class_weights
    )

    # Task 2: Localization Loss
    loss_loc = 0
    for disease_idx in range(5):
        pred_mask = outputs['localization'][disease_idx]
        gt_mask = targets['lesion_masks'][disease_idx]

        loss_dice = dice_loss(pred_mask, gt_mask)
        loss_bce = F.binary_cross_entropy(pred_mask, gt_mask)
        loss_loc += (loss_dice + loss_bce)

    loss_loc /= 5

    # Task 3: Report Generation Loss
    loss_report = outputs['report']['loss']  # Cross-entropy from LLM

    # Task weights (tunable)
    w_cls = 1.0
    w_loc = 2.0  # Higher weight for localization
    w_report = 1.5

    # Total loss
    total_loss = w_cls * loss_cls + w_loc * loss_loc + w_report * loss_report

    return {
        'total': total_loss,
        'classification': loss_cls,
        'localization': loss_loc,
        'report': loss_report
    }
```

### Training Strategy

#### Phase 1: Individual Task Pre-training
1. **Classification**: Train classification head only (5 epochs)
2. **Localization**: Train localization module only (10 epochs)
3. **Report Generation**: Vision-language alignment (5 epochs)

#### Phase 2: Joint Multi-task Fine-tuning (20 epochs)
- Train all modules simultaneously
- Gradually unfreeze vision encoder layers
- Use task weighting to balance gradients

#### Phase 3: Reinforcement Learning for Reports (5 epochs)
- Freeze classification and localization heads
- Apply GRPO/DAPO to report generation
- Use classification and localization results as context

### Gradient Balancing
- Use GradNorm or uncertainty-based weighting
- Monitor task-specific validation metrics
- Adjust task weights dynamically

## Model Configuration Summary

| Component | Model/Method | Parameters | Notes |
|-----------|-------------|------------|-------|
| Vision Encoder | ThreeDVisionEncoder | 88M | From Phase 3.1 |
| Classification Head | Text-prompt similarity | 1M | BioBERT encoder |
| Localization Module | 3-stage (Denoising + Attention U-Net) | 45M | Per-disease segmentation |
| Report Generator | Llama-3.1-70B + LoRA | 70B (325M trainable) | LoRA rank=64 |
| **Total** | | **~203M trainable** | ~70B total with frozen LLM |

## Data Requirements

### Training Data
1. **CT-RATE**: 21,907 cases
   - Classification labels: 18 diseases
   - Segmentation masks: Available for 5 key diseases
   - Reports: Not available → Use synthetic/template-based

2. **OmniAbnorm**: 10,117 cases (CoT data)
   - Classification labels: Yes
   - Detailed reasoning: Yes (from Llama-3.1-70B)
   - Use for report generation training

3. **Additional Report Data** (Recommended):
   - MIMIC-CXR: Chest X-ray reports (can adapt to CT)
   - OpenI: Radiology reports with images
   - Use for report generation pre-training

### Data Format
```python
{
    'ct_volume': np.ndarray,  # (D, H, W) or (C, D, H, W)
    'seg_mask': np.ndarray,   # (D, H, W) with region IDs

    # Classification targets
    'disease_labels': np.ndarray,  # (18,) binary labels

    # Localization targets
    'lesion_masks': {
        'pericardial_effusion': np.ndarray,  # (D, H, W) binary
        'pleural_effusion': np.ndarray,
        'consolidation': np.ndarray,
        'ground_glass': np.ndarray,
        'lung_nodule': np.ndarray,
    },

    # Report generation target
    'report_text': str,  # Clinical radiology report
    'report_sections': {
        'findings': str,
        'impression': str,
        'recommendations': str
    }
}
```

## Expected Performance

### Classification (18 diseases)
- **Metric**: AUC-ROC, Macro F1-score
- **Expected**: AUC 0.85-0.92, F1 0.75-0.85

### Localization (5 diseases)
- **Metric**: Dice score, IoU, AP50
- **Expected**: Dice 0.65-0.80, IoU 0.55-0.70

### Report Generation
- **Metrics**: BLEU-4, ROUGE-L, BERTScore, CIDEr
- **Expected**:
  - BLEU-4: 0.25-0.35
  - ROUGE-L: 0.40-0.50
  - BERTScore: 0.75-0.85
  - Clinical accuracy (manual): 80-90%

## Implementation Files Structure

```
Method/Vision_Encoder_3D/
├── model/
│   ├── __init__.py
│   ├── vision_encoder.py              # From Phase 3.1
│   ├── classification_head.py         # NEW: Multi-label classifier
│   ├── localization_module.py         # NEW: 3-stage localization
│   │   ├── TextEmbedder
│   │   ├── DenoisingTransformer
│   │   └── TextGuidedAttentionUNet
│   ├── report_generator.py            # NEW: LLM-based report generator
│   │   ├── VisionToLLMProjector
│   │   ├── LesionLocationInjector
│   │   └── ReportGeneratorWrapper
│   └── multi_task_model.py            # NEW: Unified multi-task model
│
├── training/
│   ├── __init__.py
│   ├── dataset_multitask.py           # NEW: Multi-task dataset loader
│   ├── train_multitask.py             # NEW: Multi-task training script
│   ├── losses.py                      # NEW: All loss functions
│   └── metrics.py                     # NEW: Evaluation metrics
│
├── configs/
│   └── multitask_config.yaml          # Training configuration
│
└── SUB_OBJECTIVE_3_ARCHITECTURE.md    # This file
```

## Next Steps

1. ✅ Architecture design document (this file)
2. ⬜ Implement classification head
3. ⬜ Implement localization module
4. ⬜ Implement report generator
5. ⬜ Implement multi-task integration
6. ⬜ Create training script
7. ⬜ Create comprehensive training plan document
8. ⬜ Test end-to-end pipeline
