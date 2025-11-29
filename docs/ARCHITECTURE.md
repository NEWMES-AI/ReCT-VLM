# 3D CT-Specialized Vision Encoder Architecture

## Overview

This document describes the architecture of the 3D CT-Specialized Vision Encoder for the VLM3D Challenge. The encoder is designed to:
1. Learn 3D spatial representations from CT volumes
2. Align visual features with Chain-of-Thought (CoT) reasoning
3. Incorporate anatomical segmentation information
4. Enable region-aware attention mechanisms

## Design Principles

### 1. 3D Spatial Context Preservation
- **Challenge**: CT scans are 3D volumes with continuous anatomical structures across slices
- **Solution**: Extend 2D patch-based encoding to 3D patches with slice-aware processing
- **Implementation**: 3D convolutions + slice-aware attention

### 2. Reasoning-Aligned Representation Learning
- **Challenge**: Standard vision encoders only learn image-text correspondence
- **Solution**: Align features with CoT reasoning steps, not just final descriptions
- **Implementation**: Cross-modal contrastive learning with CoT anchors

### 3. Anatomical Structure Integration
- **Challenge**: Lesion interpretation depends on anatomical location context
- **Solution**: Incorporate segmentation masks as positional embeddings
- **Implementation**: Region-aware attention with mask-guided encoding

## Architecture Components

```
┌─────────────────────────────────────────────────────────────────┐
│              3D CT-Specialized Vision Encoder                    │
└─────────────────────────────────────────────────────────────────┘

Input: CT Volume (D, H, W) + Segmentation Mask (D, H, W)
    ↓
┌─────────────────────────────────────────────────────────────────┐
│  Component 1: 3D Patch Embedding                                 │
├─────────────────────────────────────────────────────────────────┤
│  • 3D Patch Extraction (patch_size: 16×16×4)                    │
│  • Linear projection to embedding dimension                      │
│  • Add 3D positional embeddings (x, y, z)                       │
│  • Output: Patch embeddings (N_patches, D_embed)                │
└─────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────┐
│  Component 2: Anatomical Structure Encoder                       │
├─────────────────────────────────────────────────────────────────┤
│  • Encode segmentation mask to region embeddings                │
│  • Create region-to-patch mapping                               │
│  • Generate anatomical positional encodings                     │
│  • Output: Region embeddings + Patch-to-region assignments      │
└─────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────┐
│  Component 3: 3D Vision Transformer Blocks                       │
├─────────────────────────────────────────────────────────────────┤
│  Block 1-N: For each transformer block:                         │
│    ├─ Layer Norm                                                │
│    ├─ Multi-Head Self-Attention (slice-aware)                  │
│    │   • Attention bias based on z-distance                    │
│    │   • Stronger attention within same/adjacent slices        │
│    ├─ Add & Norm                                                │
│    ├─ Region-Aware Attention                                    │
│    │   • Query: Current patch features                         │
│    │   • Key/Value: Region embeddings                          │
│    │   • Weighted by patch-to-region assignments               │
│    ├─ Add & Norm                                                │
│    ├─ Feed-Forward Network                                      │
│    └─ Add & Norm                                                │
│                                                                  │
│  Output: Contextualized patch features (N_patches, D_embed)     │
└─────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────┐
│  Component 4: Multi-Scale Feature Aggregation                   │
├─────────────────────────────────────────────────────────────────┤
│  • Extract features from multiple transformer layers            │
│  • Feature Pyramid Network (FPN) for multi-scale features      │
│  • Global pooling for volume-level features                     │
│  • Region pooling for anatomical-specific features             │
│                                                                  │
│  Output:                                                         │
│  ├─ Global features: (D_embed)                                  │
│  ├─ Patch features: (N_patches, D_embed)                        │
│  └─ Region features: (N_regions, D_embed)                       │
└─────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────┐
│  Component 5: Reasoning-Aligned Projection                       │
├─────────────────────────────────────────────────────────────────┤
│  • Project visual features to shared embedding space           │
│  • Align with text encoder (from CoT reasoning)                │
│  • Multiple projection heads:                                   │
│    ├─ Global head: Volume-level alignment                      │
│    ├─ Local head: Region-level alignment                       │
│    └─ Step head: CoT step-level alignment                      │
│                                                                  │
│  Output: Aligned embeddings (D_shared)                          │
└─────────────────────────────────────────────────────────────────┘
```

## Detailed Component Specifications

### 1. 3D Patch Embedding

**Purpose**: Convert 3D CT volume to sequence of patch embeddings while preserving spatial structure.

**Implementation**:
```python
class ThreeDPatchEmbedding(nn.Module):
    def __init__(
        self,
        in_channels=1,           # CT: single channel
        embed_dim=768,
        patch_size=(4, 16, 16),  # (depth, height, width)
        img_size=(64, 512, 512)  # Typical CT volume size
    ):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size

        # 3D convolution for patch extraction
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        # 3D positional embeddings
        n_patches = (img_size[0] // patch_size[0]) * \
                    (img_size[1] // patch_size[1]) * \
                    (img_size[2] // patch_size[2])
        self.pos_embed = nn.Parameter(
            torch.zeros(1, n_patches, embed_dim)
        )

    def forward(self, x):
        # x: (B, 1, D, H, W)
        B = x.shape[0]

        # Extract patches
        x = self.proj(x)  # (B, embed_dim, D', H', W')

        # Flatten spatial dimensions
        x = x.flatten(2).transpose(1, 2)  # (B, N_patches, embed_dim)

        # Add positional embeddings
        x = x + self.pos_embed

        return x
```

**Key Features**:
- Uses 3D convolution for slice-aware feature extraction
- Maintains z-axis (slice) information in embeddings
- Positional encoding includes depth dimension

### 2. Anatomical Structure Encoder

**Purpose**: Encode segmentation masks into region-specific embeddings and create patch-to-region mappings.

**Implementation**:
```python
class AnatomicalStructureEncoder(nn.Module):
    def __init__(
        self,
        num_regions=20,          # Number of anatomical regions
        embed_dim=768,
        patch_size=(4, 16, 16)
    ):
        super().__init__()
        self.num_regions = num_regions
        self.patch_size = patch_size

        # Region embedding table
        self.region_embeds = nn.Embedding(num_regions + 1, embed_dim)

        # Learnable region features
        self.region_features = nn.Parameter(
            torch.randn(num_regions, embed_dim)
        )

        # Mask encoder (3D CNN)
        self.mask_encoder = nn.Sequential(
            nn.Conv3d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(128, embed_dim, 3, padding=1)
        )

    def forward(self, segmentation_mask, patch_coords):
        """
        Args:
            segmentation_mask: (B, 1, D, H, W) - region IDs
            patch_coords: (B, N_patches, 3) - patch centers
        Returns:
            region_embeds: (B, N_regions, embed_dim)
            patch_to_region: (B, N_patches, N_regions) - soft assignment
        """
        B, _, D, H, W = segmentation_mask.shape

        # Encode mask
        mask_features = self.mask_encoder(segmentation_mask.float())

        # Extract region embeddings from mask
        # ... (implementation details)

        # Create patch-to-region mapping
        # For each patch, compute overlap with each region
        # ... (implementation details)

        return region_embeds, patch_to_region
```

**Key Features**:
- Converts segmentation masks to learnable region embeddings
- Creates soft assignments from patches to regions
- Enables region-conditioned attention

### 3. Slice-Aware Multi-Head Attention

**Purpose**: Implement attention mechanism that considers 3D spatial relationships, particularly slice proximity.

**Implementation**:
```python
class SliceAwareAttention(nn.Module):
    def __init__(
        self,
        embed_dim=768,
        num_heads=12,
        slice_distance_weight=0.5
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.slice_distance_weight = slice_distance_weight

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)

        # Learnable slice distance bias
        max_slice_distance = 32
        self.slice_bias = nn.Parameter(
            torch.zeros(num_heads, max_slice_distance)
        )

    def forward(self, x, patch_positions):
        """
        Args:
            x: (B, N_patches, embed_dim)
            patch_positions: (B, N_patches, 3) - (z, y, x) coordinates
        """
        B, N, C = x.shape

        # Compute QKV
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Standard attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Add slice distance bias
        z_coords = patch_positions[:, :, 0]  # (B, N)
        z_dist = torch.abs(
            z_coords.unsqueeze(2) - z_coords.unsqueeze(1)
        )  # (B, N, N)

        # Apply learnable bias based on distance
        z_dist_clamped = torch.clamp(z_dist, max=31).long()
        slice_bias = self.slice_bias[:, z_dist_clamped]  # Broadcasting

        attn = attn + slice_bias * self.slice_distance_weight

        # Apply softmax and attention
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)

        return x
```

**Key Features**:
- Standard multi-head attention with slice distance bias
- Stronger attention for patches in same/adjacent slices
- Learnable bias parameters for different slice distances

### 4. Region-Aware Attention

**Purpose**: Enable patches to attend to their corresponding anatomical regions.

**Implementation**:
```python
class RegionAwareAttention(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(embed_dim, embed_dim)
        self.kv = nn.Linear(embed_dim, embed_dim * 2)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, patch_features, region_features, patch_to_region):
        """
        Args:
            patch_features: (B, N_patches, embed_dim)
            region_features: (B, N_regions, embed_dim)
            patch_to_region: (B, N_patches, N_regions) - assignment weights
        """
        B, N_patches, C = patch_features.shape
        _, N_regions, _ = region_features.shape

        # Query from patches
        q = self.q(patch_features).reshape(
            B, N_patches, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)

        # Key and Value from regions
        kv = self.kv(region_features).reshape(
            B, N_regions, 2, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        # Compute attention
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Weight attention by patch-to-region assignment
        # Expand patch_to_region for multi-head
        assignment = patch_to_region.unsqueeze(1).expand(
            -1, self.num_heads, -1, -1
        )
        attn = attn * assignment

        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N_patches, C)
        x = self.proj(x)

        return x
```

**Key Features**:
- Cross-attention between patches and anatomical regions
- Weighted by patch-to-region assignments from segmentation
- Enables patches to gather context from their anatomical location

### 5. Complete 3D Vision Encoder

**Implementation**:
```python
class ThreeDVisionEncoder(nn.Module):
    def __init__(
        self,
        in_channels=1,
        embed_dim=768,
        depth=12,
        num_heads=12,
        num_regions=20,
        patch_size=(4, 16, 16),
        img_size=(64, 512, 512)
    ):
        super().__init__()

        # Patch embedding
        self.patch_embed = ThreeDPatchEmbedding(
            in_channels, embed_dim, patch_size, img_size
        )

        # Anatomical structure encoder
        self.anatomical_encoder = AnatomicalStructureEncoder(
            num_regions, embed_dim, patch_size
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            ThreeDTransformerBlock(
                embed_dim, num_heads, num_regions
            ) for _ in range(depth)
        ])

        # Feature aggregation
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.region_pool = RegionPooling(embed_dim)

        # Projection heads
        self.global_proj = nn.Linear(embed_dim, embed_dim)
        self.local_proj = nn.Linear(embed_dim, embed_dim)
        self.step_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, ct_volume, segmentation_mask):
        """
        Args:
            ct_volume: (B, 1, D, H, W)
            segmentation_mask: (B, 1, D, H, W)
        Returns:
            global_features: (B, embed_dim)
            patch_features: (B, N_patches, embed_dim)
            region_features: (B, N_regions, embed_dim)
        """
        # Patch embedding
        x = self.patch_embed(ct_volume)  # (B, N_patches, embed_dim)

        # Get patch positions
        patch_positions = self.get_patch_positions(ct_volume.shape)

        # Encode anatomical structure
        region_embeds, patch_to_region = self.anatomical_encoder(
            segmentation_mask, patch_positions
        )

        # Transformer blocks
        for block in self.blocks:
            x = block(x, region_embeds, patch_to_region, patch_positions)

        # Aggregate features
        global_features = self.global_pool(x.transpose(1, 2)).squeeze(-1)
        region_features = self.region_pool(x, patch_to_region, region_embeds)

        # Project to shared space
        global_proj = self.global_proj(global_features)
        local_proj = self.local_proj(x)
        step_proj = self.step_proj(region_features)

        return {
            'global': global_proj,
            'local': local_proj,
            'region': step_proj,
            'patch_features': x
        }
```

## Training Strategy

### 1. Cross-Modal Contrastive Learning

**Objective**: Align visual features with CoT reasoning at multiple granularities.

**Implementation**:
```python
class CrossModalContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, visual_features, text_features):
        """
        Args:
            visual_features: (B, D)
            text_features: (B, D)
        """
        # Normalize
        visual_features = F.normalize(visual_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        # Compute similarity
        logits = torch.matmul(
            visual_features, text_features.T
        ) / self.temperature

        # Labels: positive pairs on diagonal
        labels = torch.arange(len(visual_features)).to(logits.device)

        # Bidirectional loss
        loss_v2t = self.criterion(logits, labels)
        loss_t2v = self.criterion(logits.T, labels)

        return (loss_v2t + loss_t2v) / 2
```

**Multi-Level Alignment**:
1. **Global alignment**: Volume-level features ↔ Final impression
2. **Region alignment**: Region-specific features ↔ Anatomical mentions in CoT
3. **Step alignment**: Patch features ↔ Individual CoT reasoning steps

### 2. CoT-Anchored Learning

**Purpose**: Use CoT steps as anchors to guide visual attention.

**Strategy**:
- For each CoT step mentioning a specific anatomical region
- Extract corresponding region features
- Align with text embedding of that step
- Enforce that patch attention focuses on mentioned regions

### 3. Multi-Task Training

**Objectives**:
1. **Contrastive learning**: Align vision and text
2. **Segmentation auxiliary task**: Predict anatomical regions
3. **CoT generation**: Generate reasoning from visual features

## Data Processing Pipeline

### Input Data Format

**OmniAbnorm CoT**:
```json
{
  "case_id": "973_1_0_177",
  "image_path": "image/973_1_0_177/0.jpeg",
  "mask_path": "mask/973_1_0_177/0.jpeg",
  "region": "Chest",
  "category": "pulmonary thin-walled cavitation",
  "cot_reasoning": [
    {
      "step": 1,
      "text": "Lung Parenchyma - Bilateral cysts...",
      "anatomical_entities": [...],
      "mask_region": "Bilateral hemithoraces"
    }
  ]
}
```

**CT-RATE CoT**:
```json
{
  "volumename": "train_18451_a_2.nii.gz",
  "question": "Given the findings...",
  "answer": "The radiology report indicates..."
}
```

### Preprocessing

1. **CT Volume Loading**:
   - Load .nii.gz or .npz format
   - Apply windowing (lung window: center=40, width=400)
   - Resize to fixed shape (64, 512, 512)
   - Normalize to [0, 1]

2. **Segmentation Mask Loading**:
   - Load corresponding mask
   - Map to anatomical region IDs
   - Resize to match CT volume

3. **Text Processing**:
   - Extract CoT steps
   - Tokenize using text encoder (BERT/BioBERT)
   - Create step-to-region mappings

## Evaluation Metrics

1. **Retrieval Performance**:
   - Image-to-text retrieval (Recall@k)
   - Text-to-image retrieval (Recall@k)

2. **Alignment Quality**:
   - Region-text alignment accuracy
   - Step-patch attention correlation

3. **Downstream Tasks**:
   - Lesion classification accuracy
   - Lesion localization IoU
   - Report generation BLEU/ROUGE scores

## Implementation Plan

### Phase 3.1: Core Architecture
- [ ] Implement ThreeDPatchEmbedding
- [ ] Implement SliceAwareAttention
- [ ] Implement AnatomicalStructureEncoder
- [ ] Implement RegionAwareAttention
- [ ] Implement complete ThreeDVisionEncoder

### Phase 3.2: Training Infrastructure
- [ ] Dataset loader for CT-RATE + OmniAbnorm
- [ ] Text encoder (BioBERT)
- [ ] Cross-modal contrastive loss
- [ ] Training loop with multi-GPU support

### Phase 3.3: Evaluation & Refinement
- [ ] Retrieval evaluation
- [ ] Attention visualization
- [ ] CoT-alignment validation
- [ ] Hyperparameter tuning

## File Structure

```
Method/Vision_Encoder_3D/
├── ARCHITECTURE.md                    # This file
├── model/
│   ├── patch_embedding.py             # 3D patch embedding
│   ├── anatomical_encoder.py          # Anatomical structure encoder
│   ├── attention.py                   # Slice-aware & region-aware attention
│   ├── transformer_block.py           # Complete transformer block
│   └── vision_encoder.py              # Complete 3D vision encoder
├── loss/
│   ├── contrastive.py                 # Cross-modal contrastive loss
│   └── multi_task.py                  # Multi-task loss functions
├── data/
│   ├── dataset.py                     # Dataset loader
│   ├── preprocessing.py               # CT volume preprocessing
│   └── text_processing.py             # CoT text processing
├── training/
│   ├── train.py                       # Training script
│   ├── trainer.py                     # Trainer class
│   └── config.yaml                    # Training configuration
└── evaluation/
    ├── retrieval.py                   # Retrieval evaluation
    ├── visualization.py               # Attention visualization
    └── metrics.py                     # Evaluation metrics
```

## References

1. **CT-CLIP**: CT-CLIP: A CT Image and Report Contrastive Learning Pre-training Method
2. **MedSAM2**: Segment Anything in Medical Images
3. **ViT**: An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
4. **CLIP**: Learning Transferable Visual Models From Natural Language Supervision
5. **BiomedCLIP**: Large-Scale Domain-Specific Pretraining for Biomedical Vision-Language Processing
