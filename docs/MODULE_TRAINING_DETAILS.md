# 모듈별 학습 상세 정보

## Overview

이 문서는 VLM3D Multi-task System의 각 모듈별 학습 세부사항을 정리합니다.
각 모듈에 대해 다음 정보를 포함합니다:
- Epoch 수
- 데이터 유형
- 파일 포맷
- 학습 데이터 수
- 파일 용량
- 예상 학습 시간

---

## Module 1: 3D Vision Encoder

### 모델 정보
- **모델명**: ThreeDVisionEncoder (Custom Implementation)
- **기반 모델**: Vision Transformer (ViT) architecture adapted for 3D medical imaging
- **주요 컴포넌트**:
  - Patch Embedding: 3D Conv (kernel: 4×16×16)
  - Transformer Blocks: 12 layers, 12 attention heads
  - Attention Mechanisms: SliceAwareAttention, RegionAwareAttention
  - Anatomical Encoder: AnatomicalStructureEncoder (20 regions)
- **파라미터 수**: 88M
- **구현 위치**: `model/vision_encoder.py`
- **연구 내용**: Native 3D processing with slice-aware and region-aware attention for CT volumes

### 학습 설정

| 항목 | 내용 |
|------|------|
| **Phase** | Pre-training (Vision encoder only) |
| **Epoch 수** | 50 epochs |
| **Batch Size** | 32 (16 per GPU × 2 GPUs) |
| **Total Iterations** | 34,250 iterations (685 iterations/epoch) |
| **Learning Rate** | 5e-5 (base), 3e-5 (vision) |

### 데이터 정보

#### 데이터 유형
```
입력 데이터:
1. CT Volume: 3D medical imaging data
   - Windowed CT scans (lung window: center=40, width=400)
   - Native 3D structure preserved

2. Segmentation Mask: Anatomical structure labels
   - 20개 해부학적 영역 (organs, vessels, bones)
   - Object ID로 라벨링 (1-20)
   - MedSAM2 기반 생성

출력 데이터:
- Global features: (B, 768) - 전체 볼륨 representation
- Local features: (B, 16384, 768) - Patch-level features
- Region features: (B, 20, 768) - Anatomical region features
```

#### 파일 포맷
```python
# NPZ 파일 구조
{
    'imgs': np.ndarray,      # (D, H, W) uint8 [0-255]
                             # Typical: (64, 512, 512)
                             # CT volume after windowing

    'gts': np.ndarray,       # (D, H, W) uint8
                             # Segmentation mask with object IDs
                             # 0 = background, 1-20 = organs/structures

    'spacing': np.ndarray    # (3,) float32
                             # Physical voxel spacing in mm
                             # Format: [z_spacing, y_spacing, x_spacing]
                             # Typical: [2.0, 0.742, 0.742]
}

# 파일명 예시
train_1000_a_1.npz    # train: split, 1000: case_id, a: sub_id, 1: slice_id
valid_50_b_2.npz
```

#### 학습 데이터 수
```
Training Set:
├── CT-RATE Dataset
│   ├── Total cases: 21,907 cases
│   ├── Lung nodule subset: ~21,000 cases (main focus)
│   ├── Per case: 1 volume + 1 mask
│   └── Total files: 21,907 NPZ files

Validation Set:
└── CT-RATE Dataset
    ├── Total cases: 1,000 cases (subset for faster validation)
    ├── Per case: 1 volume + 1 mask
    └── Total files: 1,000 NPZ files

Total: 22,907 cases
```

#### 파일 용량
```
Individual File:
├── CT Volume (64×512×512 uint8): ~16 MB
├── Segmentation Mask (64×512×512 uint8): ~16 MB
├── Metadata: <1 KB
└── Total per NPZ: ~20-50 MB (depending on compression)

Dataset Total:
├── Training Set: 21,907 files × 32 MB (avg) = ~701 GB (raw)
│                                              ~263 GB (compressed NPZ)
├── Validation Set: 1,000 files × 32 MB = ~32 GB (raw)
│                                          ~12 GB (compressed)
└── Total Dataset: ~275 GB (compressed NPZ format)

Storage Recommendation: 500 GB free space (including working directory)
```

### 학습 시간 예상
```
Hardware: 2× NVIDIA H200 (80GB VRAM each)

Per Iteration:
├── Data loading: 0.15 sec
├── Forward pass: 0.35 sec
├── Backward pass: 0.25 sec
└── Optimizer step: 0.06 sec
Total: ~0.81 sec/iteration

Per Epoch:
├── Training: 685 iterations × 0.81 sec = 555 sec (9.25 min)
├── Validation: ~2 min (every 5 epochs)
└── Epoch total: ~9.5 min

Total Training:
├── 50 epochs × 9.5 min = 475 min
├── Validation overhead: ~20 min
├── Checkpointing: ~10 min
└── Total: ~8.4 hours

With overhead and warmup: 10-12 hours
```

---

## Module 2: Multi-label Classification Head

### 모델 정보
- **모델명**: MultiLabelClassifier (Custom Implementation)
- **텍스트 인코더**:
  - **Primary**: `emilyalsentzer/Bio_ClinicalBERT` (BioBERT, 110M params, frozen)
  - **Alternative**: `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`
  - HuggingFace Transformers 기반
- **주요 컴포넌트**:
  - Text Encoder: BioBERT/ClinicalBERT (frozen)
  - Vision Projector: Linear layers (768 → 512)
  - Text Projector: Linear layers (768 → 512)
  - Similarity Computation: Cosine similarity with temperature scaling
  - Learnable Components: Soft prompts (18 diseases), temperature, bias
- **파라미터 수**: 111M total (1M trainable, 110M BioBERT frozen)
- **구현 위치**: `model/classification_head.py`
- **연구 내용**: Text-prompt based multi-label disease classification using vision-text similarity

### 학습 설정

| 항목 | 내용 |
|------|------|
| **Phase** | Phase 1.1 - Classification only |
| **Epoch 수** | 10 epochs |
| **Batch Size** | 64 (32 per GPU × 2 GPUs, gradient accumulation 2) |
| **Total Iterations** | 3,420 iterations (342 iterations/epoch) |
| **Learning Rate** | 5e-5 |

### 데이터 정보

#### 데이터 유형
```
입력 데이터:
1. Vision Features: From 3D Vision Encoder
   - Global features: (B, 768)
   - Region features: (B, 20, 768) [optional]
   - Pre-extracted or computed on-the-fly

2. Disease Labels: Multi-label binary classification
   - 18 major thoracic diseases
   - Binary labels per disease (0 or 1)
   - Multiple diseases can co-occur

질병 목록 (18개):
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

출력 데이터:
- Classification logits: (B, 18) float32
- Probabilities: (B, 18) float32 [0-1] after sigmoid
- Predictions: (B, 18) int64 [0 or 1] after thresholding
```

#### 파일 포맷
```python
# CSV 형식 (CT-RATE predicted labels)
{
    'VolumeName': 'train_1000_a_1',              # Volume identifier
    'Lung nodule': 1,                            # Binary label
    'Pleural effusion': 0,
    'Consolidation': 1,
    ...                                          # 18 disease columns
}

# JSON 형식 (OmniAbnorm CoT data)
{
    "image_id": "case_00001",
    "image_path": "images/case_00001.jpg",
    "findings": [
        {
            "disease": "lung_nodule",
            "present": true,
            "confidence": 0.92,
            "location": "right upper lobe"
        },
        {
            "disease": "pleural_effusion",
            "present": true,
            "confidence": 0.87,
            "location": "bilateral"
        }
    ],
    "reasoning": "Step 1: Observed round opacity in right upper lobe...",
    "summary": "Multiple findings including nodule and effusion",
    "generated_by": "llama-3.1-70b",
    "timestamp": "2025-11-29T10:30:00Z"
}

# NumPy 형식 (학습 시 사용)
{
    'volume_id': str,              # Volume identifier
    'features': np.ndarray,        # (768,) or (20, 768)
    'labels': np.ndarray,          # (18,) binary labels
    'weights': np.ndarray          # (18,) class weights [optional]
}
```

#### 학습 데이터 수
```
Dataset 1: CT-RATE (Main training data)
├── Training samples: 21,907 cases
├── Validation samples: 1,000 cases
├── Label format: Predicted labels from CT-RATE
├── Label source: train_predicted_labels.csv, valid_predicted_labels.csv
├── Per sample: 1 volume → 18 binary labels
└── Total labels: 21,907 × 18 = 394,326 label entries

Dataset 2: OmniAbnorm (CoT data)
├── Training samples: 10,117 cases (1,315 현재 생성, 나머지 진행 중)
├── Validation samples: ~500 cases (5% split)
├── Label format: Structured findings with reasoning
├── Per sample: 1 image → multiple findings + CoT reasoning
└── Total: ~10,117 samples with detailed annotations

Combined Training Data:
├── Total unique cases: ~32,000 (CT-RATE + OmniAbnorm)
├── CT-RATE: Primary 3D data with labels
├── OmniAbnorm: Supplementary with detailed reasoning
└── Usage: CT-RATE for main training, OmniAbnorm for CoT alignment

Class Distribution (CT-RATE):
├── Lung nodule: ~21,000 positive (~95%)
├── Pleural effusion: ~8,000 positive (~36%)
├── Consolidation: ~5,000 positive (~23%)
├── Ground-glass: ~3,500 positive (~16%)
├── Other diseases: varying (1-30%)
└── Note: Highly imbalanced → need class weighting
```

#### 파일 용량
```
CT-RATE Labels:
├── CSV file (train_predicted_labels.csv): ~8 MB
│   ├── 21,907 rows × 20 columns (ID + 18 diseases + metadata)
│   └── Text format, highly compressible
├── CSV file (valid_predicted_labels.csv): ~400 KB
└── Total: ~8.5 MB

OmniAbnorm Data:
├── JSON files: ~1,315 files × 5 KB = ~6.5 MB (current)
│   └── Target: ~10,117 files × 5 KB = ~50 MB (complete)
├── Images: ~10,117 images × 100 KB (avg) = ~1 GB
│   └── JPG format, various resolutions
└── Total: ~1.05 GB (complete dataset)

Pre-extracted Features (optional, for faster training):
├── If pre-extracting vision features:
│   ├── Global features: 21,907 × 768 × 4 bytes = ~67 MB
│   ├── Region features: 21,907 × 20 × 768 × 4 bytes = ~1.3 GB
│   └── Total: ~1.4 GB (saves forward pass time)
├── Trade-off: Disk space vs computation time
└── Recommendation: Compute on-the-fly for flexibility

Text Embeddings (cached):
├── Disease prompts: 18 diseases × 512 dim × 4 bytes = ~36 KB
├── BioBERT weights: ~440 MB (loaded in memory, not saved)
└── Total: Negligible disk usage

Total Storage (Classification Module):
├── Labels (CT-RATE): 8.5 MB
├── CoT Data (OmniAbnorm): 1.05 GB
├── Optional pre-extracted features: 1.4 GB
└── Total: ~1.1 GB (without pre-extraction)
           ~2.5 GB (with pre-extraction)
```

### 학습 시간 예상
```
Hardware: 2× NVIDIA H200

Per Iteration:
├── Data loading: 0.05 sec
├── Forward pass (vision encoder): 0.2 sec
├── Forward pass (classification): 0.05 sec
├── Loss computation: 0.02 sec
├── Backward pass: 0.15 sec
└── Optimizer step: 0.03 sec
Total: ~0.5 sec/iteration

Per Epoch:
├── Training: 342 iterations × 0.5 sec = 171 sec (2.85 min)
├── Validation: ~30 sec (every 2 epochs)
└── Epoch total: ~3 min

Total Training (Phase 1.1):
├── 10 epochs × 3 min = 30 min
├── Validation overhead: ~5 min
├── Checkpointing: ~2 min
└── Total: ~37 minutes

실제 예상: 30-40 minutes
```

---

## Module 3: Lesion Localization Module (3-stage)

### 모델 정보
- **모델명**: LesionLocalizationModule (Custom 3-stage Architecture)
- **텍스트 인코더**:
  - **Model**: `emilyalsentzer/Bio_ClinicalBERT` (BioBERT, 110M params, frozen)
  - 5개 질병별 상세 anatomical description 인코딩
- **Stage 1: TextEmbedder**
  - BioBERT 기반 disease text embedding
  - Pre-computed disease embeddings (5 diseases)
- **Stage 2: DenoisingTransformer**
  - 4-layer transformer with cross-attention
  - Text-guided feature refinement
  - Parameters: ~25M
- **Stage 3: TextGuidedAttentionUNet**
  - U-Net architecture with attention gates
  - Text-modulated skip connections
  - Base channels: 64 (configurable: 32/48/64)
  - Parameters: ~20M
- **파라미터 수**: 45M trainable (Denoising 25M + U-Net 20M) + 110M BioBERT (frozen)
- **구현 위치**: `model/localization_module.py`
- **연구 내용**: Text-guided 3-stage localization (Text → Denoising → Attention U-Net)

### 학습 설정

| 항목 | 내용 |
|------|------|
| **Phase** | Phase 1.2 - Localization only |
| **Epoch 수** | 15 epochs |
| **Batch Size** | 64 (8 per GPU × 2 GPUs, gradient accumulation 8) |
| **Total Iterations** | 5,130 iterations (342 iterations/epoch) |
| **Learning Rate** | 3e-5 |

### 데이터 정보

#### 데이터 유형
```
입력 데이터:
1. Vision Features: From 3D Vision Encoder
   - Local patch features: (B, N_patches, 768)
     - N_patches = 16,384 for (64, 512, 512) volume with patch size (4, 16, 16)
   - Region features: (B, 20, 768) [optional]

2. Disease Text: Disease-specific anatomical descriptions
   - 5개 주요 질병별 상세 prompt
   - BioBERT/ClinicalBERT로 인코딩
   - Pre-computed embeddings (768 dim)

3. Original Shape: Volume dimensions for upsampling
   - Typical: (64, 512, 512) - (D, H, W)

Target Diseases (5개):
1. Pericardial effusion (심낭 삼출)
2. Pleural effusion (흉막 삼출)
3. Consolidation (경화)
4. Ground-glass opacity (간유리 음영)
5. Lung nodules (폐결절)

출력 데이터:
- Segmentation masks: Dict of (B, D, H, W) per disease
  - Float values [0-1] (logits after sigmoid)
  - Binary masks after thresholding (>0.5)
- Per disease: Separate high-resolution segmentation map
```

#### 파일 포맷
```python
# Input: Vision features (computed on-the-fly or pre-extracted)
{
    'volume_id': 'train_1000_a_1',
    'local_features': np.ndarray,      # (N_patches, 768)
    'region_features': np.ndarray,     # (20, 768)
    'original_shape': tuple            # (64, 512, 512)
}

# Target: Segmentation masks (from CT-RATE)
{
    'volume_id': 'train_1000_a_1',

    # Mask files (separate .nii.gz per disease)
    'masks': {
        'pericardial_effusion': np.ndarray,    # (D, H, W) binary
        'pleural_effusion': np.ndarray,        # (D, H, W) binary
        'consolidation': np.ndarray,           # (D, H, W) binary
        'ground_glass_opacity': np.ndarray,    # (D, H, W) binary
        'lung_nodule': np.ndarray              # (D, H, W) binary
    }
}

# NIfTI 파일 구조 (.nii.gz)
NIfTI Header:
├── Dimensions: (64, 512, 512)
├── Voxel spacing: (2.0, 0.742, 0.742) mm
├── Data type: uint8 or int16
├── Orientation: RAS or LPS
└── Affine matrix: 4×4 transformation

Data:
├── Shape: (D, H, W) = (64, 512, 512)
├── Values:
│   ├── 0: Background (no lesion)
│   ├── 1: Lesion present (for binary masks)
│   └── 1-N: Multiple lesion IDs (for instance segmentation)
└── Storage: Compressed with gzip

# 파일명 예시
train_1000_a_1_pericardial_effusion_mask.nii.gz
train_1000_a_1_pleural_effusion_mask.nii.gz
```

#### 학습 데이터 수
```
CT-RATE Segmentation Masks:
├── Total volumes with masks: 21,907 cases
├── Diseases per volume: 5 diseases
├── Mask availability: ~75% of volumes have at least 1 disease mask
│   ├── Pericardial effusion: ~2,000 cases (~9%)
│   ├── Pleural effusion: ~8,000 cases (~36%)
│   ├── Consolidation: ~5,000 cases (~23%)
│   ├── Ground-glass: ~3,500 cases (~16%)
│   └── Lung nodule: ~21,000 cases (~95%)
└── Total mask files: ~40,000 mask files across all diseases

Training Strategy:
├── Positive samples (with lesions): ~16,000 volumes
│   └── Used for localization training
├── Negative samples (no lesions): ~5,900 volumes
│   └── Used for false positive reduction
└── Validation: 1,000 volumes with masks

Per-Disease Training Data:
┌────────────────────────┬──────────┬───────────┬──────────┐
│ Disease                │ Train    │ Valid     │ Total    │
├────────────────────────┼──────────┼───────────┼──────────┤
│ Pericardial effusion   │ ~1,900   │ ~100      │ ~2,000   │
│ Pleural effusion       │ ~7,600   │ ~400      │ ~8,000   │
│ Consolidation          │ ~4,750   │ ~250      │ ~5,000   │
│ Ground-glass opacity   │ ~3,300   │ ~200      │ ~3,500   │
│ Lung nodule            │ ~20,000  │ ~1,000    │ ~21,000  │
└────────────────────────┴──────────┴───────────┴──────────┘

Label Statistics:
├── Average lesions per positive case: 1-3 lesions
├── Lesion size distribution:
│   ├── Small (<10 voxels): 30%
│   ├── Medium (10-1000 voxels): 50%
│   └── Large (>1000 voxels): 20%
└── Multi-label co-occurrence: ~40% have multiple lesion types
```

#### 파일 용량
```
Individual Mask File:
├── Dimensions: 64 × 512 × 512 = 16,777,216 voxels
├── Data type: uint8 (1 byte per voxel)
├── Raw size: ~16 MB
├── Compressed (.nii.gz): ~1-5 MB (depending on lesion density)
│   ├── Sparse masks (few lesions): ~1 MB
│   ├── Dense masks (many lesions): ~5 MB
│   └── Average: ~2.5 MB
└── With header and metadata: ~2.5 MB per file

Dataset Total Size:
├── Pericardial effusion: 2,000 files × 2.5 MB = ~5 GB
├── Pleural effusion: 8,000 files × 2.5 MB = ~20 GB
├── Consolidation: 5,000 files × 2.5 MB = ~12.5 GB
├── Ground-glass: 3,500 files × 2.5 MB = ~8.75 GB
├── Lung nodule: 21,000 files × 2.5 MB = ~52.5 GB
└── Total segmentation masks: ~99 GB

Storage Breakdown:
├── Original CT volumes: 263 GB (from Module 1)
├── Segmentation masks: ~99 GB (lesion masks)
├── Anatomical masks: ~12 GB (organ segmentation, included above)
└── Total (CT + Masks): ~362 GB

Recommended Storage: 500 GB free space

Pre-extracted Features (optional):
├── If pre-extracting local features:
│   ├── Local features: 21,907 × 16,384 × 768 × 4 bytes = ~2.2 TB
│   └── Too large → Compute on-the-fly recommended
├── Cache denoised features per epoch:
│   ├── Denoised: 21,907 × 16,384 × 768 × 4 bytes = ~2.2 TB
│   └── Also too large → Compute on-the-fly
└── Recommendation: No pre-extraction, compute dynamically
```

### 학습 시간 예상
```
Hardware: 2× NVIDIA H200

Per Iteration (single disease):
├── Data loading: 0.1 sec
├── Forward pass:
│   ├── Vision encoder: 0.3 sec
│   ├── Text embedder: <0.01 sec (pre-computed)
│   ├── Denoising transformer: 0.4 sec
│   └── Attention U-Net: 0.6 sec
│   Total forward: 1.3 sec
├── Loss computation (Dice + BCE): 0.05 sec
├── Backward pass: 0.8 sec
└── Optimizer step: 0.04 sec
Total: ~2.3 sec/iteration (single disease)

Per Iteration (all 5 diseases):
├── Can process in parallel or sequentially
├── Sequential: 2.3 × 5 = ~11.5 sec
├── With optimization: ~1.5 sec/iteration
│   └── Shared feature computation, parallel U-Nets
└── Practical: ~1.5 sec/iteration

Per Epoch:
├── Training: 342 iterations × 1.5 sec = 513 sec (8.55 min)
├── Validation: ~3 min (every 3 epochs)
└── Epoch total: ~9 min

Total Training (Phase 1.2):
├── 15 epochs × 9 min = 135 min (2.25 hours)
├── Validation overhead: ~15 min
├── Checkpointing: ~5 min
└── Total: ~2.5 hours

실제 예상: 2-3 hours
```

---

## Module 4: Report Generation Module

### 모델 정보
- **모델명**: ReportGenerator (Custom Vision-LLM Integration)
- **LLM 백본 (선택 가능)**:
  - **Configuration A (Large)**: `meta-llama/Llama-3.1-70B-Instruct` (70B params)
    - Recommended for best quality
    - 4-bit quantization: ~35 GB VRAM
    - LoRA trainable: ~325M params (rank=64)
  - **Configuration B (Medium)**: `google/gemma-2-27b-it` (27B params)
    - Balanced performance/resource
    - 4-bit quantization: ~14 GB VRAM
    - LoRA trainable: ~120M params (rank=64)
  - **Configuration C (Small)**: `meta-llama/Llama-3.1-8B-Instruct` (8B params)
    - Fast experimentation
    - 4-bit quantization: ~4.5 GB VRAM
    - LoRA trainable: ~40M params (rank=32)
- **주요 컴포넌트**:
  - **VisionToLLMProjector**: Q-Former style cross-attention
    - Learnable query tokens: 256 (default)
    - Cross-attention layers: 3
    - Output dimension: 4096 (LLM embedding space)
    - Parameters: ~256M
  - **LesionLocationInjector**: Text-based location extraction
    - Automatic anatomical location parsing from masks
    - Text prompt injection strategy
  - **LoRA Adapters**: Parameter-efficient fine-tuning
    - Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
    - Rank: 64 (Large/Medium), 32 (Small)
    - Alpha: 128
- **파라미터 수**: 70.3B total (325M trainable via LoRA) for Configuration A
- **구현 위치**: `model/report_generator.py`
- **연구 내용**: LLM-based clinical report generation with vision-language alignment

### 학습 설정

| 항목 | Phase 1.3 (Alignment) | Phase 2 (Joint) | Phase 3 (RL) |
|------|-----------------------|-----------------|--------------|
| **Epoch 수** | 5 epochs | 40 epochs | 10 epochs |
| **Batch Size** | 64 (effective) | 64 (effective) | 64 (effective) |
| **Total Iterations** | 1,710 | 13,680 | 3,420 |
| **Learning Rate** | 1e-4 (proj), 1e-5 (LoRA) | 5e-5 (proj), 1e-5 (LoRA) | 5e-6 (LoRA only) |
| **목적** | Vision-LLM alignment | Multi-task joint | Report quality |

### 데이터 정보

#### 데이터 유형
```
입력 데이터:
1. Vision Features: From 3D Vision Encoder
   - Global features: (B, 768)
   - Local features: (B, N_patches, 768)
   - Region features: (B, 20, 768)
   - Projected to LLM space: (B, num_visual_tokens, 4096)
     - num_visual_tokens: 256 (default)

2. Lesion Information: From Localization Module
   - Segmentation masks: Dict of (B, D, H, W) per disease
   - Location descriptions: List of text strings
     - Example: "[FINDING] Pleural effusion detected in right lower posterior region"

3. Classification Results: From Classification Module
   - Disease predictions: (B, 18) probabilities
   - Detected diseases: List of disease names per sample

4. Text Prompt: Instruction for report generation
   - Default: "Generate a clinical radiology report based on the CT scan findings:"
   - Can be customized for specific report sections

출력 데이터:
- Generated report text: String format
- Structured sections:
  ├── FINDINGS: Detailed description of abnormalities
  ├── IMPRESSION: Summary and clinical significance
  └── RECOMMENDATIONS: Follow-up suggestions [optional]

Report Format:
"""
[FINDING] Pleural effusion detected in right lower posterior region
[FINDING] Lung nodule detected in upper left anterior region

Generate a clinical radiology report based on the CT scan findings:

FINDINGS:
- Right-sided pleural effusion is present in the lower posterior region,
  suggesting fluid accumulation in the pleural space.
- A small lung nodule measuring approximately 8mm is identified in the
  left upper lobe anterior segment.
- No consolidation or ground-glass opacities are observed.

IMPRESSION:
Right pleural effusion and left upper lobe pulmonary nodule.
Clinical correlation and follow-up imaging recommended.

RECOMMENDATIONS:
- Follow-up CT in 3-6 months to assess nodule stability
- Clinical correlation for cause of pleural effusion
"""
```

#### 파일 포맷
```python
# Training Data Format 1: Synthetic Reports (from CT-RATE)
{
    'volume_id': 'train_1000_a_1',
    'classification_results': {
        'diseases': ['pleural_effusion', 'lung_nodule'],
        'probabilities': [0.87, 0.92, ...]  # 18 values
    },
    'localization_results': {
        'pleural_effusion': {
            'mask': np.ndarray,  # (D, H, W)
            'location': 'right lower posterior region',
            'size': 'moderate'
        },
        'lung_nodule': {
            'mask': np.ndarray,
            'location': 'left upper anterior region',
            'size': '8mm'
        }
    },
    'report': {
        'full_text': "FINDINGS: ...\n\nIMPRESSION: ...",
        'sections': {
            'findings': "Detailed findings...",
            'impression': "Summary and significance...",
            'recommendations': "Follow-up suggestions..."
        },
        'generation_method': 'template-based',  # or 'llm-generated'
        'quality_score': 0.75  # if available
    }
}

# JSON Format (Report Dataset)
{
    "report_id": "report_001",
    "volume_id": "train_1000_a_1",
    "report_text": "FINDINGS:\n- Right pleural effusion...",
    "findings_list": [
        "Right pleural effusion present",
        "Small lung nodule in left upper lobe"
    ],
    "diseases_mentioned": [
        "pleural_effusion",
        "lung_nodule"
    ],
    "report_length": 156,  # words
    "sections": {
        "findings": "...",
        "impression": "...",
        "recommendations": "..."
    },
    "metadata": {
        "generated_date": "2025-11-29",
        "reviewer": "template",
        "quality": "synthetic"
    }
}

# Text Format (Simple Reports)
# File: reports/train_1000_a_1.txt
"""
FINDINGS:
- Right-sided pleural effusion is present in the lower posterior region.
- A small lung nodule measuring approximately 8mm is identified in the
  left upper lobe anterior segment.
- No consolidation or ground-glass opacities are observed.

IMPRESSION:
Right pleural effusion and left upper lobe pulmonary nodule.
Clinical correlation and follow-up imaging recommended.

RECOMMENDATIONS:
- Follow-up CT in 3-6 months to assess nodule stability
- Clinical correlation for cause of pleural effusion
"""

# Tokenized Format (for training)
{
    'input_ids': torch.LongTensor,      # (seq_len,) tokenized prompt + report
    'attention_mask': torch.LongTensor,  # (seq_len,) attention mask
    'labels': torch.LongTensor,          # (seq_len,) labels for training
    'visual_tokens': torch.FloatTensor   # (256, 4096) projected vision features
}
```

#### 학습 데이터 수
```
Dataset 1: CT-RATE with Synthetic Reports
├── Training samples: 21,907 reports
│   ├── Generated from classification + localization results
│   ├── Template-based generation
│   ├── Quality: Adequate for initial training
│   └── Per report: 50-200 words average
├── Validation samples: 1,000 reports
└── Total: 22,907 synthetic reports

Synthetic Report Generation Strategy:
├── Method 1: Template-based (80%)
│   ├── Use predefined templates per disease
│   ├── Fill in locations from localization
│   ├── Add severity based on mask size
│   └── Example: "FINDINGS: {disease} detected in {location}..."
│
├── Method 2: LLM-generated (20%)
│   ├── Use Llama-3.1-70B to generate diverse reports
│   ├── Input: Classification + localization results
│   ├── Output: Natural clinical report
│   └── Higher quality but more computational cost
│
└── Combination: Mix both for diversity

Dataset 2: Real Reports (Optional, for quality improvement)
├── Source: MIMIC-CXR (Chest X-ray reports)
│   ├── Total: ~227,000 reports
│   ├── Can adapt to CT domain
│   ├── High-quality clinical language
│   └── Requires domain adaptation
│
├── Source: OpenI (Radiology reports)
│   ├── Total: ~3,600 reports
│   ├── Mixed modalities (X-ray, CT)
│   ├── Publicly available
│   └── Smaller but diverse
│
└── Usage: Fine-tune after synthetic pre-training

Dataset 3: OmniAbnorm CoT (Supplementary)
├── Samples: 10,117 cases with reasoning
├── Format: Detailed finding descriptions + reasoning chains
├── Use: Enhance report quality with reasoning
│   ├── Not full reports, but detailed descriptions
│   ├── Can be used as auxiliary training data
│   └── Improves clinical reasoning in reports
└── Integration: Add to training as supplementary data

Training Data Summary:
┌─────────────────────────┬──────────┬───────────┬──────────┐
│ Dataset                 │ Train    │ Valid     │ Total    │
├─────────────────────────┼──────────┼───────────┼──────────┤
│ CT-RATE Synthetic       │ 21,907   │ 1,000     │ 22,907   │
│ OmniAbnorm Descriptions │ 10,117   │ 500       │ 10,617   │
│ Optional: MIMIC-CXR     │ 200,000  │ 27,000    │ 227,000  │
│ Optional: OpenI         │ 3,200    │ 400       │ 3,600    │
└─────────────────────────┴──────────┴───────────┴──────────┘

Practical Training Set:
├── Phase 1.3 (Alignment): CT-RATE Synthetic (21,907)
├── Phase 2 (Joint): CT-RATE Synthetic (21,907)
├── Phase 3 (RL): CT-RATE Synthetic (21,907)
└── Total unique reports: 21,907

Report Statistics:
├── Average length: 120 words (50-200 range)
├── Average tokens (Llama tokenizer): ~180 tokens
├── Sections per report:
│   ├── FINDINGS: 70-120 words
│   ├── IMPRESSION: 20-40 words
│   └── RECOMMENDATIONS: 15-30 words [optional]
└── Diseases mentioned per report: 1-4 (average 2.3)
```

#### 파일 용량
```
Text Reports:
├── Individual report size:
│   ├── Average: 120 words × 6 chars/word = 720 chars
│   ├── Text size: ~720 bytes
│   └── With metadata (JSON): ~2 KB
│
├── CT-RATE Synthetic Reports:
│   ├── 21,907 reports × 2 KB = ~43.8 MB
│   └── Compressed: ~30 MB
│
├── OmniAbnorm Descriptions:
│   ├── 10,117 files × 5 KB = ~50 MB
│   └── Already counted in Module 2
│
└── Total: ~50 MB (CT-RATE reports)

Tokenized Data (pre-processed):
├── If pre-tokenizing for faster training:
│   ├── Per sample:
│   │   ├── Input IDs: 512 tokens × 4 bytes = 2 KB
│   │   ├── Attention mask: 512 tokens × 4 bytes = 2 KB
│   │   ├── Labels: 512 tokens × 4 bytes = 2 KB
│   │   └── Total: ~6 KB per sample
│   │
│   ├── Full dataset:
│   │   └── 21,907 samples × 6 KB = ~131 MB
│   │
│   └── Recommendation: Pre-tokenize for faster data loading
│
└── Total with tokenization: ~180 MB

LLM Model Weights:
├── Llama-3.1-70B (4-bit quantized):
│   ├── Base model: ~35 GB (loaded in GPU memory)
│   ├── On disk: ~38 GB
│   └── LoRA adapters: ~280 MB (saved per checkpoint)
│
├── Alternative: Gemma-2-27B (4-bit):
│   ├── Base model: ~14 GB
│   ├── On disk: ~15 GB
│   └── LoRA adapters: ~120 MB
│
└── Alternative: Llama-3.1-8B (4-bit):
    ├── Base model: ~4.5 GB
    ├── On disk: ~5 GB
    └── LoRA adapters: ~40 MB

Vision-to-LLM Projector:
├── Model weights: 256M parameters × 4 bytes = ~1 GB
├── Saved checkpoints: ~1 GB per checkpoint
└── Negligible compared to LLM

Total Storage (Report Generation):
├── Report text data: 50 MB
├── Tokenized data: 180 MB
├── LLM model (Llama-70B): 38 GB (downloaded once)
├── Vision projector: 1 GB
└── Total: ~40 GB (mostly LLM)

Checkpoint Storage:
├── Per checkpoint:
│   ├── LoRA adapters: 280 MB
│   ├── Vision projector: 1 GB
│   ├── Optimizer states: 500 MB
│   └── Total: ~1.8 GB per checkpoint
│
├── Keep 3 best checkpoints: ~5.4 GB
└── Total with checkpoints: ~45 GB

Recommended Storage: 60 GB free for report generation module
```

### 학습 시간 예상

#### Phase 1.3: Vision-LLM Alignment (5 epochs)
```
Hardware: 2× NVIDIA H200
Config: Freeze LLM, train projector only

Per Iteration:
├── Data loading: 0.1 sec
├── Forward pass:
│   ├── Vision encoder: 0.3 sec (frozen, but still compute)
│   ├── Vision projector: 0.15 sec
│   ├── LLM forward: 1.2 sec (70B model, even frozen is slow)
│   └── Total forward: 1.65 sec
├── Loss computation: 0.05 sec
├── Backward pass (projector only): 0.2 sec
└── Optimizer step: 0.05 sec
Total: ~2.0 sec/iteration

Per Epoch:
├── Training: 342 iterations × 2.0 sec = 684 sec (11.4 min)
├── Validation: ~5 min (every 2 epochs)
└── Epoch total: ~12 min

Total Training (Phase 1.3):
├── 5 epochs × 12 min = 60 min
├── Validation overhead: ~10 min
├── Checkpointing: ~3 min
└── Total: ~73 minutes (1.2 hours)

실제 예상: 1-1.5 hours
```

#### Phase 2: Joint Multi-task Training (40 epochs)
```
Hardware: 2× NVIDIA H200
Config: All tasks, LoRA enabled

Per Iteration:
├── Data loading: 0.15 sec
├── Forward pass:
│   ├── Vision encoder: 0.35 sec
│   ├── Classification: 0.05 sec
│   ├── Localization: 1.0 sec
│   ├── Report generation: 1.5 sec
│   └── Total forward: 2.9 sec
├── Multi-task loss: 0.08 sec
├── Backward pass (all gradients): 1.3 sec
└── Optimizer step: 0.12 sec
Total: ~2.5 sec/iteration

Per Epoch:
├── Training: 342 iterations × 2.5 sec = 855 sec (14.25 min)
├── Validation: ~8 min (every 2 epochs)
└── Epoch total: ~15 min

Total Training (Phase 2):
├── 40 epochs × 15 min = 600 min (10 hours)
├── Validation overhead: ~160 min (2.7 hours)
├── Checkpointing: ~20 min
└── Total: ~12.7 hours

실제 예상: 10-12 hours
```

#### Phase 3: RL Fine-tuning (10 epochs)
```
Hardware: 2× NVIDIA H200
Config: GRPO with 4 generations per sample

Per Iteration:
├── Data loading: 0.1 sec
├── Forward pass (K=4 generations):
│   ├── Vision encoding: 0.3 sec (once)
│   ├── LLM generation #1: 3.0 sec (512 tokens)
│   ├── LLM generation #2: 3.0 sec
│   ├── LLM generation #3: 3.0 sec
│   ├── LLM generation #4: 3.0 sec
│   └── Total generation: 12.3 sec
├── Reward computation:
│   ├── BLEU/ROUGE: 0.2 sec (all 4)
│   ├── BERTScore: 1.5 sec (optional, expensive)
│   ├── Clinical accuracy: 0.3 sec
│   └── Total reward: 2.0 sec
├── Policy gradient loss: 0.1 sec
├── Backward pass: 0.4 sec
└── Optimizer step: 0.05 sec
Total: ~15 sec/iteration (with BERTScore)
      ~13.5 sec/iteration (without BERTScore)

Per Epoch:
├── Training: 342 iterations × 13.5 sec = 4,617 sec (77 min)
├── Validation: ~10 min (every 2 epochs)
└── Epoch total: ~80 min

Total Training (Phase 3):
├── 10 epochs × 80 min = 800 min (13.3 hours)
├── Validation overhead: ~50 min
├── Checkpointing: ~10 min
└── Total: ~14.3 hours

실제 예상: 12-15 hours

Optimization:
├── Skip BERTScore: Save ~40% time → 8-10 hours
├── Reduce K to 2: Save ~50% time → 6-8 hours
├── Both: → 4-5 hours
└── Recommended: K=2, skip BERTScore for Phase 3
```

---

## Module 5: Multi-task Integration (Phase 2)

### 모델 정보
- **모델명**: VLM3DMultiTask (Unified Multi-task System)
- **통합 아키텍처**: All modules from Phase 1 + Shared vision encoder
- **구성 요소**:
  - **Shared Backbone**: ThreeDVisionEncoder (88M, from Module 1)
  - **Task 1**: MultiLabelClassifier (1M trainable + BioBERT frozen, from Module 2)
    - Text model: `emilyalsentzer/Bio_ClinicalBERT`
  - **Task 2**: LesionLocalizationModule (45M, from Module 3)
    - Text model: `emilyalsentzer/Bio_ClinicalBERT` (shared with Task 1)
    - 3-stage: TextEmbedder → DenoisingTransformer → AttentionUNet
  - **Task 3**: ReportGenerator (325M trainable, from Module 4)
    - LLM: `meta-llama/Llama-3.1-70B-Instruct` (Configuration A)
    - Alternative: `google/gemma-2-27b-it` or `meta-llama/Llama-3.1-8B-Instruct`
    - LoRA rank: 64
- **Multi-task Loss**:
  - Classification: Binary Cross-Entropy with class weights
  - Localization: Dice Loss + BCE (per disease)
  - Generation: Cross-Entropy (from LLM)
  - Weighted combination: w_cls=1.0, w_loc=2.0, w_gen=1.5
- **파라미터 수**:
  - Total: 70.5B (with Llama-70B)
  - Trainable: ~460M (Vision 88M + Classification 1M + Localization 45M + LoRA 325M)
  - Frozen: ~70B (BioBERT 110M + Llama-70B base)
- **구현 위치**: `model/multi_task_model.py`
- **연구 내용**: Joint multi-task learning for classification, localization, and report generation with shared 3D vision encoder

### 학습 설정

| 항목 | 내용 |
|------|------|
| **Phase** | Phase 2 - Joint multi-task training |
| **Epoch 수** | 40 epochs |
| **Batch Size** | 64 (8 per GPU × 2 GPUs, gradient accumulation 8) |
| **Total Iterations** | 13,680 iterations (342 iterations/epoch) |
| **Learning Rates** | Vision: 1e-5, Tasks: 3e-5, LoRA: 1e-5 |

### 데이터 정보

#### 데이터 유형
```
Multi-task Training Sample:
{
    # Inputs
    'ct_volume': np.ndarray,            # (1, D, H, W) or (D, H, W)
    'segmentation_mask': np.ndarray,    # (D, H, W) anatomical regions

    # Task 1: Classification targets
    'disease_labels': np.ndarray,       # (18,) binary labels

    # Task 2: Localization targets
    'lesion_masks': {
        'pericardial_effusion': np.ndarray,   # (D, H, W)
        'pleural_effusion': np.ndarray,
        'consolidation': np.ndarray,
        'ground_glass_opacity': np.ndarray,
        'lung_nodule': np.ndarray
    },

    # Task 3: Report generation target
    'report_text': str,                 # Full report text
    'report_tokens': torch.LongTensor,  # Tokenized report

    # Metadata
    'volume_id': str,
    'spacing': np.ndarray,              # (3,) voxel spacing
    'original_shape': tuple             # (D, H, W)
}

Output Format:
{
    # Task 1: Classification
    'classification': {
        'logits': torch.FloatTensor,         # (B, 18)
        'probabilities': torch.FloatTensor,  # (B, 18)
        'predictions': torch.LongTensor      # (B, 18)
    },

    # Task 2: Localization
    'localization': {
        'pericardial_effusion': torch.FloatTensor,  # (B, D, H, W)
        'pleural_effusion': torch.FloatTensor,
        'consolidation': torch.FloatTensor,
        'ground_glass_opacity': torch.FloatTensor,
        'lung_nodule': torch.FloatTensor
    },

    # Task 3: Report Generation
    'generation': {
        'loss': torch.FloatTensor,      # Scalar
        'logits': torch.FloatTensor,    # (B, seq_len, vocab_size)
        'generated_text': List[str]     # During inference
    },

    # Shared features
    'vision_features': {
        'global': torch.FloatTensor,    # (B, 768)
        'local': torch.FloatTensor,     # (B, N_patches, 768)
        'region': torch.FloatTensor     # (B, 20, 768)
    }
}
```

#### 파일 포맷
```
Training data combines all previous modules:

Directory Structure:
DATA/CT-RATE/
├── lung_nodule_medsam2/           # Prepared NPZ files
│   ├── train_1000_a_1.npz        # CT volume + anatomical mask
│   ├── train_1001_a_1.npz
│   └── ... (21,907 files)
│
├── lesion_masks/                  # Disease-specific masks
│   ├── pericardial_effusion/
│   │   ├── train_1000_a_1.nii.gz
│   │   └── ...
│   ├── pleural_effusion/
│   ├── consolidation/
│   ├── ground_glass_opacity/
│   └── lung_nodule/
│
├── train_predicted_labels.csv     # Classification labels
│
└── reports/                       # Generated reports
    ├── train_1000_a_1.txt
    └── ... (21,907 files)

Data Loading:
1. Load NPZ (volume + anatomical mask)
2. Load classification labels from CSV
3. Load lesion masks from corresponding disease folders
4. Load report text
5. Apply transforms
6. Collate into batch
```

#### 학습 데이터 수
```
Unified Dataset:
├── Training samples: 21,907 cases
│   ├── All have: CT volume + anatomical mask
│   ├── All have: Classification labels (18 diseases)
│   ├── ~75% have: At least 1 lesion mask (16,000 cases)
│   └── All have: Synthetic reports
│
├── Validation samples: 1,000 cases
│   └── Same structure as training
│
└── Total: 22,907 cases with complete multi-task labels

Data Availability per Task:
┌────────────────────┬─────────┬─────────┬──────────┐
│ Task               │ Train   │ Valid   │ Total    │
├────────────────────┼─────────┼─────────┼──────────┤
│ Classification     │ 21,907  │ 1,000   │ 22,907   │
│ Localization       │ 16,000  │ 750     │ 16,750   │
│ Report Generation  │ 21,907  │ 1,000   │ 22,907   │
└────────────────────┴─────────┴─────────┴──────────┘

Task Sampling Strategy:
├── Option 1: Always train all tasks (if localization mask available)
│   ├── Samples with masks: Train all 3 tasks
│   └── Samples without masks: Train classification + generation only
│
├── Option 2: Task-weighted sampling
│   ├── Weight by inverse frequency
│   ├── Oversample rare diseases for localization
│   └── Balance task difficulty
│
└── Recommended: Option 1 with dynamic task weights
```

#### 파일 용량
```
Complete Multi-task Dataset:
├── CT volumes (NPZ): 263 GB
├── Lesion masks: 99 GB
├── Classification labels (CSV): 8.5 MB
├── Reports (text): 50 MB
└── Total: ~362 GB

Memory Requirements:
├── Single sample in memory:
│   ├── CT volume: 64×512×512 × 4 bytes = 64 MB (float32)
│   ├── Anatomical mask: 64×512×512 × 1 byte = 16 MB
│   ├── Lesion masks (5): 5 × 16 MB = 80 MB
│   ├── Labels: 18 × 4 bytes = 72 bytes
│   ├── Report tokens: 512 × 4 bytes = 2 KB
│   └── Total: ~160 MB per sample
│
├── Batch (8 samples):
│   ├── Input data: 8 × 160 MB = ~1.3 GB
│   ├── Model activations: ~15 GB
│   ├── Gradients: ~5 GB
│   └── Total: ~21 GB per batch
│
└── GPU memory usage:
    ├── Model: 60 GB (with 4-bit LLM)
    ├── Batch + activations: 21 GB
    └── Total: ~81 GB (fits in 2× H200)

Storage Recommendations:
├── Raw dataset: 400 GB
├── Working directory: 50 GB
├── Checkpoints: 20 GB
├── Logs and outputs: 30 GB
└── Total needed: 500 GB free space
```

### 학습 시간 예상
```
Phase 2: Joint Multi-task Training
Hardware: 2× NVIDIA H200
Duration: 40 epochs

Per Iteration Breakdown:
├── Data loading and preprocessing:
│   ├── Load CT volume + masks: 0.08 sec
│   ├── Load labels + report: 0.02 sec
│   ├── Apply transforms: 0.03 sec
│   └── Collate batch: 0.02 sec
│   Total data: 0.15 sec
│
├── Forward pass:
│   ├── Vision encoder: 0.35 sec
│   │   ├── Patch embedding: 0.05 sec
│   │   ├── Transformer blocks: 0.25 sec
│   │   └── Feature pooling: 0.05 sec
│   │
│   ├── Classification head: 0.05 sec
│   │   ├── Text encoder (cached): <0.01 sec
│   │   ├── Vision projection: 0.02 sec
│   │   └── Similarity computation: 0.03 sec
│   │
│   ├── Localization module: 1.0 sec
│   │   ├── Text embedder (cached): <0.01 sec
│   │   ├── Denoising transformer: 0.35 sec
│   │   ├── U-Net (5 diseases): 0.6 sec
│   │   └── Upsampling: 0.05 sec
│   │
│   ├── Report generator: 1.5 sec
│   │   ├── Vision projector: 0.15 sec
│   │   ├── Location injection: 0.05 sec
│   │   ├── LLM forward (teacher forcing): 1.25 sec
│   │   └── Loss computation: 0.05 sec
│   │
│   └── Total forward: 2.9 sec
│
├── Loss computation:
│   ├── Classification loss: 0.02 sec
│   ├── Localization loss (5 diseases): 0.04 sec
│   ├── Generation loss: <0.01 sec (from forward)
│   └── Multi-task weighted sum: 0.02 sec
│   Total loss: 0.08 sec
│
├── Backward pass:
│   ├── Report generator: 0.5 sec
│   ├── Localization module: 0.4 sec
│   ├── Classification head: 0.05 sec
│   ├── Vision encoder: 0.3 sec
│   └── Gradient accumulation: 0.05 sec
│   Total backward: 1.3 sec
│
└── Optimizer step:
    ├── Parameter updates: 0.1 sec
    ├── Learning rate scheduling: <0.01 sec
    └── Total optimizer: 0.12 sec

Total per iteration: 2.5 sec

Per Epoch:
├── Training:
│   ├── 342 iterations × 2.5 sec = 855 sec
│   ├── Time: 14.25 minutes
│   └── Throughput: 24 samples/sec
│
├── Validation (every 2 epochs):
│   ├── 1,000 samples / 8 batch size = 125 iterations
│   ├── Forward only: 125 × 2.0 sec = 250 sec
│   ├── Metric computation: 3 minutes
│   └── Total validation: ~8 minutes
│
└── Epoch total: ~15 minutes (training epochs)
                 ~23 minutes (validation epochs)

Total Training Time:
├── Training epochs: 40 × 15 min = 600 min
├── Validation epochs: 20 × 8 min = 160 min (every 2 epochs)
├── Checkpointing: 8 × 2 min = 16 min (every 5 epochs)
├── Warmup and overhead: ~30 min
└── Total: ~806 minutes = 13.4 hours

With realistic overhead: 12-15 hours

Optimization Opportunities:
├── Mixed precision (enabled): Already ~30% faster
├── Gradient checkpointing: 40% memory, 15% slower
├── Data prefetching (enabled): ~10% faster
├── Compiled models (torch.compile): ~20% faster
└── Optimized: 10-12 hours possible
```

---

## Summary Table: All Modules

| Module | Model Name | Epochs | Data Type | File Format | Training Samples | Storage | Training Time |
|--------|------------|--------|-----------|-------------|------------------|---------|---------------|
| **1. Vision Encoder** | ThreeDVisionEncoder (Custom) | 50 | CT volumes + masks | NPZ | 21,907 | 275 GB | 10-12 hrs |
| **2. Classification** | MultiLabelClassifier + BioBERT | 10 | Vision features + labels | CSV/NPZ | 21,907 | 8.5 MB | 30-40 min |
| **3. Localization** | LesionLocalizationModule (3-stage) + BioBERT | 15 | Features + lesion masks | NIfTI + NPZ | 16,000 | 99 GB | 2-3 hrs |
| **4. Report Gen (P1.3)** | VisionToLLMProjector + Llama-70B (LoRA) | 5 | Features + reports | TXT/JSON | 21,907 | 50 MB | 1-1.5 hrs |
| **4. Report Gen (P3)** | Same + GRPO | 10 | Features + reports | TXT/JSON | 21,907 | 50 MB | 12-15 hrs |
| **5. Multi-task (P2)** | VLM3DMultiTask (All above integrated) | 40 | All combined | Multi-format | 21,907 | 362 GB | 12-15 hrs |
| **Total Training** | **All Components** | **80** | **Multi-modal** | **Various** | **21,907** | **~400 GB** | **~24-28 hrs** |

### Model Name Details

| Module | Primary Model | Text Encoder | LLM Backbone | Parameters (Trainable) |
|--------|---------------|--------------|--------------|----------------------|
| **Vision Encoder** | ThreeDVisionEncoder (ViT-based) | - | - | 88M |
| **Classification** | MultiLabelClassifier | `emilyalsentzer/Bio_ClinicalBERT` | - | 1M (+ 110M frozen) |
| **Localization** | LesionLocalizationModule | `emilyalsentzer/Bio_ClinicalBERT` | - | 45M (+ 110M frozen) |
| **Report Generator** | VisionToLLMProjector + LoRA | - | `meta-llama/Llama-3.1-70B-Instruct` | 325M (+ 70B frozen) |
| **Multi-task** | VLM3DMultiTask | BioBERT (shared) | Llama-70B | ~460M (+ 70B frozen) |

### Alternative Model Configurations

For different resource requirements:

| Configuration | LLM Model | Total Params | Trainable | VRAM | Training Time |
|---------------|-----------|--------------|-----------|------|---------------|
| **A (Large)** | `meta-llama/Llama-3.1-70B-Instruct` | 70.5B | 460M | ~80 GB | 24-28 hrs |
| **B (Medium)** | `google/gemma-2-27b-it` | 27.3B | 280M | ~50 GB | 18-22 hrs |
| **C (Small)** | `meta-llama/Llama-3.1-8B-Instruct` | 8.3B | 120M | ~25 GB | 12-16 hrs |

## Notes

### Data Preparation Requirements
```
Before Training:
1. Download CT-RATE dataset from HuggingFace
2. Prepare NPZ files using prepare_ctrate_for_medsam2.py
3. Generate segmentation masks (if not available)
4. Create synthetic reports
5. Split into train/validation sets

Estimated preparation time: 4-6 hours
```

### Hardware Recommendations
```
Minimum:
├── GPUs: 2× NVIDIA H200 (80GB)
├── RAM: 256 GB
├── Storage: 500 GB SSD
└── Network: High-speed for data loading

Recommended:
├── GPUs: 4× NVIDIA H200 (80GB)
├── RAM: 512 GB
├── Storage: 1 TB NVMe SSD (RAID)
└── Network: 10 GbE for distributed training
```

### Cost Estimation
```
Cloud Computing (AWS p5.48xlarge, 8× H200):
├── Instance cost: ~$98/hour
├── Training time: 28 hours
├── Total compute: ~$2,744
├── Storage (1 TB EBS): ~$100/month
└── Total estimated: ~$2,900

Alternative (smaller configuration):
├── Use Llama-8B instead of Llama-70B
├── Reduce to 2× A100 (80GB)
├── Instance cost: ~$8/hour
├── Training time: 40 hours
└── Total: ~$320
```

---

## 모델명 및 소스 요약

### 사용된 주요 모델

**자체 구현 모델 (Custom)**:
1. `ThreeDVisionEncoder` - 3D native vision transformer
2. `MultiLabelClassifier` - Text-prompt based classifier
3. `LesionLocalizationModule` - 3-stage localization pipeline
4. `VisionToLLMProjector` - Q-Former style vision-to-LLM bridge
5. `VLM3DMultiTask` - Unified multi-task system

**외부 Pre-trained 모델 (HuggingFace)**:
1. `emilyalsentzer/Bio_ClinicalBERT` - Medical domain text encoder (110M params)
   - Used in: Classification, Localization
   - Status: Frozen during training

2. `meta-llama/Llama-3.1-70B-Instruct` - Large language model (70B params)
   - Used in: Report Generation
   - Training method: LoRA (rank=64, ~325M trainable)
   - Alternative: `google/gemma-2-27b-it` or `meta-llama/Llama-3.1-8B-Instruct`

### 모델 파라미터 분포

```
Total System Parameters: ~70.5B
├── Trainable: ~460M (0.65%)
│   ├── Vision Encoder: 88M
│   ├── Classification: 1M
│   ├── Localization: 45M
│   └── Report Gen (LoRA): 325M
│
└── Frozen: ~70.2B (99.35%)
    ├── BioBERT: 110M
    └── Llama-70B: 70B
```

이제 각 모듈별로 사용된 **구체적인 모델명**과 **학습 상세 정보**가 완벽하게 정리되었습니다!