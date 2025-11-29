# VLM3D Multi-task System Training Plan

## Overview

This document provides a comprehensive training plan for the complete VLM3D multi-task learning system (Sub-objective 3), which performs:
1. **Multi-label Disease Classification** (18 diseases)
2. **Lesion Localization** (5 key diseases)
3. **Clinical Report Generation** (using LLMs)

All three tasks are trained simultaneously using a shared 3D vision encoder with task-specific heads.

## Model Information

### Model Name
**VLM3D-MultiTask: Reasoning-Aligned 3D Vision-Language Model for CT Image Understanding**

### Research Content
A multi-task learning framework that combines:
- 3D native vision encoder with anatomical context
- Text-prompt based multi-label classification
- 3-stage lesion localization with text guidance
- LLM-based clinical report generation
- Multi-task learning with shared representations

### Model Architecture Summary

| Component | Architecture | Parameters | Trainable |
|-----------|-------------|------------|-----------|
| **3D Vision Encoder** | 12-layer transformer with slice/region-aware attention | 88M | Yes (after warmup) |
| **Classification Head** | Text-prompt similarity (BioBERT) | 110M + 1M learnable | 1M (BioBERT frozen) |
| **Localization Module** | 3-stage (Text + Denoising + Attention U-Net) | 45M | Yes |
| **Report Generator** | Llama-3.1-70B with LoRA | 70B (325M LoRA) | 325M (base frozen) |
| **Text Encoders** | BioBERT/ClinicalBERT | 110M | Frozen |
| **Total Parameters** | | ~70.4B | ~460M trainable |
| **Effective Training Size** | | | **~460M parameters** |

### Model Configurations

We provide three model size configurations:

#### Configuration A: Large (Recommended for Best Performance)
- **LLM**: Llama-3.1-70B-Instruct
- **Vision Encoder**: 12 layers, 768 dim
- **Localization U-Net**: 64 base channels
- **Total Trainable**: ~460M
- **Memory**: ~60GB (4-bit quantization)
- **Training Speed**: ~2.5 iterations/sec (2× H200)

#### Configuration B: Medium (Balanced)
- **LLM**: Gemma-2-27B-it
- **Vision Encoder**: 12 layers, 768 dim
- **Localization U-Net**: 48 base channels
- **Total Trainable**: ~280M
- **Memory**: ~32GB (4-bit quantization)
- **Training Speed**: ~3.5 iterations/sec (2× H200)

#### Configuration C: Small (Fast Experimentation)
- **LLM**: Llama-3.1-8B-Instruct
- **Vision Encoder**: 8 layers, 512 dim
- **Localization U-Net**: 32 base channels
- **Total Trainable**: ~120M
- **Memory**: ~16GB (4-bit quantization)
- **Training Speed**: ~5 iterations/sec (2× H200)

**This plan uses Configuration A (Large) for maximum performance.**

## Training Data

### Dataset Sources

#### 1. CT-RATE Dataset
- **Source**: HuggingFace `ibrahimhamamci/CT-RATE`
- **Description**: Large-scale CT scan dataset with predicted labels and segmentation masks
- **Files**: `.nii.gz` format (volumes and masks)

**Data Statistics:**
- Training samples: 21,907 volumes
- Validation samples: 1,000 volumes (subset for faster evaluation)
- Lung nodule cases: ~21,000 (focus subset)

**Available Labels:**
- Classification: 18 major diseases (from predicted labels CSV)
- Segmentation: Masks available for 5 key diseases
  - Pericardial effusion
  - Pleural effusion
  - Consolidation
  - Ground-glass opacity
  - Lung nodules

**File Format:**
```python
CT volume: (D, H, W) uint8, windowed [0-255]
Segmentation mask: (D, H, W) uint8, object IDs
Typical size: ~20-50 MB per case
```

**Total Size:** ~263 GB

#### 2. OmniAbnorm CoT Dataset
- **Source**: Generated using Llama-3.1-70B (Phase 2)
- **Description**: Chain-of-Thought reasoning data for classification
- **Files**: `.json` format with image paths and CoT text

**Data Statistics:**
- Current: 1,315 cases with detailed reasoning
- Target: 10,117 cases (in progress)
- Images: `.jpg` format from dataset

**Available Data:**
- Classification labels with confidence scores
- Step-by-step reasoning chains
- Detailed finding descriptions

**File Format:**
```json
{
  "image_id": "...",
  "findings": ["disease1", "disease2"],
  "reasoning": "Step 1: ... Step 2: ...",
  "confidence": 0.85
}
```

**Total Size:** ~1 GB (complete dataset)

#### 3. Synthetic Reports (Generated)
Since CT-RATE doesn't include clinical reports, we synthesize them from available data:

**Generation Strategy:**
- Template-based generation from classification + localization results
- Example templates:
  ```
  FINDINGS:
  - {disease_name} detected in {location} with {severity}
  - ...

  IMPRESSION:
  {summary_of_key_findings}
  ```

**Statistics:**
- Training reports: 21,907 (one per CT-RATE training case)
- Report length: 50-200 words average
- Quality: Template-based (adequate for initial training)

**Future Enhancement:**
- Real radiology reports from MIMIC-CXR (chest X-ray, can adapt)
- Manual report annotation for validation set

**Total Size:** ~50 MB (text only)

### Combined Dataset Summary

| Dataset | Samples | Tasks | Format | Size |
|---------|---------|-------|--------|------|
| CT-RATE (train) | 21,907 | Classification, Localization | .nii.gz | 263 GB |
| CT-RATE (valid) | 1,000 | Classification, Localization | .nii.gz | 12 GB |
| OmniAbnorm | 10,117 | Classification (CoT) | .jpg + .json | 1 GB |
| Synthetic Reports | 21,907 | Report Generation | .txt | 50 MB |
| **Total** | **~32,000 training samples** | **All tasks** | **Mixed** | **~276 GB** |

### Data Splits

```
Training Set:
├── CT-RATE: 21,907 cases
│   ├── Classification: Yes (18 diseases)
│   ├── Localization: Yes (5 diseases)
│   └── Reports: Synthetic
└── OmniAbnorm: 10,117 cases
    ├── Classification: Yes with CoT
    ├── Localization: No
    └── Reports: No

Validation Set:
└── CT-RATE: 1,000 cases
    ├── Classification: Yes
    ├── Localization: Yes
    └── Reports: Synthetic

Test Set:
└── VLM3D Challenge (MICCAI 2025)
    └── Held-out test set (size TBD)
```

## Training Configuration

### Hardware Requirements

**Minimum:**
- 2× NVIDIA H200 GPUs (80GB VRAM each)
- 256GB System RAM
- 2TB NVMe SSD (for dataset)
- 100GB+ free space (checkpoints, logs)

**Recommended:**
- 4× NVIDIA H200 GPUs (for faster training)
- 512GB System RAM
- 4TB NVMe RAID (for I/O speed)

**Our Setup:**
- 2× NVIDIA H200 (80GB VRAM)
- 512GB RAM
- 4TB NVMe SSD

### Training Hyperparameters

#### Phase 1: Task-Specific Pre-training (30 epochs)

Each task is pre-trained individually to establish good initialization:

**1.1. Classification Only (10 epochs)**
```yaml
optimizer: AdamW
learning_rate: 5e-5
weight_decay: 0.01
batch_size: 16 per GPU (32 total)
gradient_accumulation: 2 (effective batch 64)
warmup_steps: 500
lr_schedule: cosine with linear warmup
mixed_precision: bfloat16
enabled_tasks: [classification]
freeze_vision_encoder: False
```

**Iterations per epoch**: 21,907 / 64 = 342 iterations
**Total iterations**: 342 × 10 = 3,420
**Time per iteration**: ~0.5 sec
**Epoch time**: ~3 minutes
**Phase time**: **~30 minutes**

**1.2. Localization Only (15 epochs)**
```yaml
optimizer: AdamW
learning_rate: 3e-5
weight_decay: 0.01
batch_size: 4 per GPU (8 total)  # Larger memory for U-Net
gradient_accumulation: 8 (effective batch 64)
warmup_steps: 500
lr_schedule: cosine
mixed_precision: bfloat16
enabled_tasks: [localization]
freeze_vision_encoder: False
```

**Iterations per epoch**: 21,907 / 64 = 342 iterations
**Total iterations**: 342 × 15 = 5,130
**Time per iteration**: ~1.5 sec (U-Net forward + backward)
**Epoch time**: ~9 minutes
**Phase time**: **~2.2 hours**

**1.3. Report Generation Alignment (5 epochs)**
```yaml
optimizer: AdamW
learning_rate: 1e-4 (projector), 1e-5 (LoRA)
weight_decay: 0.01
batch_size: 4 per GPU (8 total)  # LLM memory intensive
gradient_accumulation: 8 (effective batch 64)
warmup_steps: 200
lr_schedule: cosine
mixed_precision: bfloat16
enabled_tasks: [generation]
freeze_vision_encoder: True
freeze_llm: True (except LoRA)
```

**Iterations per epoch**: 21,907 / 64 = 342 iterations
**Total iterations**: 342 × 5 = 1,710
**Time per iteration**: ~2.0 sec (LLM forward pass)
**Epoch time**: ~11 minutes
**Phase time**: **~55 minutes**

**Phase 1 Total Time: ~3.5 hours**

#### Phase 2: Joint Multi-task Fine-tuning (40 epochs)

All tasks trained simultaneously with shared vision encoder:

```yaml
optimizer: AdamW
learning_rates:
  vision_encoder: 1e-5
  classification_head: 3e-5
  localization_module: 3e-5
  report_projector: 5e-5
  lora_adapters: 1e-5

weight_decay: 0.01
batch_size: 4 per GPU (8 total)
gradient_accumulation: 8 (effective batch 64)
warmup_epochs: 2
lr_schedule: cosine with restarts
mixed_precision: bfloat16

enabled_tasks: [classification, localization, generation]

# Task loss weights
task_weights:
  classification: 1.0
  localization: 2.0  # Higher weight for harder task
  generation: 1.5

# Task sampling (if needed to balance)
task_sampling: uniform  # or weighted by difficulty

# Gradient clipping
max_grad_norm: 1.0

# EMA (optional)
use_ema: True
ema_decay: 0.9999
```

**Iterations per epoch**: 21,907 / 64 = 342 iterations
**Total iterations**: 342 × 40 = 13,680
**Time per iteration**: ~2.5 sec (all tasks + multi-task loss)
**Epoch time**: ~14 minutes
**Phase time**: **~9.5 hours**

#### Phase 3: Reinforcement Learning for Reports (Optional, 10 epochs)

Fine-tune report generation with GRPO/DAPO for style and quality:

```yaml
method: GRPO (Group Relative Policy Optimization)
learning_rate: 5e-6
batch_size: 2 per GPU (4 total)
num_generations_per_sample: 4  # K=4 for GRPO
gradient_accumulation: 16 (effective batch 64)

reward_weights:
  clinical_accuracy: 0.4
  factual_consistency: 0.3
  fluency: 0.2
  format_adherence: 0.1

enabled_tasks: [generation]
freeze_all_except_lora: True
```

**Iterations per epoch**: 21,907 / 64 = 342 iterations
**Total iterations**: 342 × 10 = 3,420
**Time per iteration**: ~8.0 sec (4 generations + reward computation)
**Epoch time**: ~45 minutes
**Phase time**: **~7.5 hours**

### Total Training Time Estimation

| Phase | Epochs | Iterations | Time/Iter | Total Time |
|-------|--------|------------|-----------|------------|
| **Phase 1.1: Classification** | 10 | 3,420 | 0.5s | 30 min |
| **Phase 1.2: Localization** | 15 | 5,130 | 1.5s | 2.2 hrs |
| **Phase 1.3: Report Alignment** | 5 | 1,710 | 2.0s | 55 min |
| **Phase 2: Joint Training** | 40 | 13,680 | 2.5s | 9.5 hrs |
| **Phase 3: RL for Reports** | 10 | 3,420 | 8.0s | 7.5 hrs |
| **Total** | **80 epochs** | **27,360 iters** | **~2.9s avg** | **~20.5 hours** |

**With validation, checkpointing, and overhead: ~24-28 hours total**

### Checkpoint Strategy

```yaml
checkpoint_frequency: every 5 epochs
keep_last_n: 3
keep_best_n: 2  # Based on validation metrics
checkpoint_metrics:
  - validation_total_loss
  - validation_dice_mean
  - validation_auc_macro

checkpoint_size: ~2.5 GB per checkpoint
total_storage_needed: ~15 GB (6 checkpoints)
```

### Monitoring and Logging

```yaml
logging:
  tensorboard: True
  wandb: True (optional)
  log_frequency: every 10 iterations

validation:
  frequency: every 2 epochs
  num_samples: 1,000 (full validation set)
  tasks: [classification, localization]  # Skip generation for speed

metrics_tracked:
  classification:
    - AUC-ROC (macro, micro)
    - F1 Score (macro, micro)
    - Precision, Recall
  localization:
    - Dice coefficient (mean, per-disease)
    - IoU score
  generation:
    - BLEU-4
    - ROUGE-L
    - Perplexity

visualization:
  frequency: every 5 epochs
  num_samples: 5
  save_segmentation_overlays: True
  save_generated_reports: True
```

## Training Pipeline

### Data Loading Pipeline

```python
# Pseudocode
class MultiTaskDataset:
    def __init__(self, ct_rate_path, omniabnorm_path, report_path):
        # Load CT-RATE volumes and masks
        self.ct_volumes = load_ct_rate_volumes()
        self.ct_labels = load_ct_rate_labels()
        self.ct_masks = load_ct_rate_masks()

        # Load OmniAbnorm
        self.omniabnorm_data = load_omniabnorm()

        # Load/generate reports
        self.reports = load_or_generate_reports()

    def __getitem__(self, idx):
        # Load CT volume
        ct_volume = self.ct_volumes[idx]  # (D, H, W)
        seg_mask = self.seg_masks[idx]    # (D, H, W)

        # Get all task labels
        class_labels = self.ct_labels[idx]  # (18,)
        lesion_masks = self.ct_masks[idx]   # dict of (D, H, W)
        report_text = self.reports[idx]      # str

        # Apply transforms
        ct_volume, seg_mask, lesion_masks = self.transform(
            ct_volume, seg_mask, lesion_masks
        )

        # Tokenize report
        report_tokens = self.tokenizer(report_text)

        return {
            'ct_volume': ct_volume,
            'segmentation_mask': seg_mask,
            'disease_labels': class_labels,
            'lesion_masks': lesion_masks,
            'report_tokens': report_tokens,
            'report_text': report_text
        }

# Data augmentation
transforms = Compose([
    RandomFlip3D(p=0.5),
    RandomRotation3D(degrees=15, p=0.3),
    RandomElasticDeformation3D(p=0.2),
    IntensityShift(shift=0.1, p=0.3),
    RandomGaussianNoise(std=0.01, p=0.2)
])

# DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    prefetch_factor=2
)
```

### Training Loop

```python
# Pseudocode
def train_multi_task(model, train_loader, val_loader, config):
    # Setup
    optimizer = AdamW(model.parameters(), lr=config.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)
    loss_fn = MultiTaskLoss(**config.loss_config)
    metrics = MultiTaskMetrics()

    # Training loop
    for epoch in range(config.epochs):
        model.train()
        epoch_losses = defaultdict(float)

        for batch_idx, batch in enumerate(train_loader):
            # Forward pass
            outputs = model(
                ct_volume=batch['ct_volume'],
                segmentation_mask=batch['segmentation_mask'],
                disease_labels=batch['disease_labels'],
                lesion_masks_gt=batch['lesion_masks'],
                report_tokens=batch['report_tokens']
            )

            # Compute losses
            losses = loss_fn(outputs, batch)
            loss = losses['total']

            # Backward pass
            loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % config.gradient_accumulation == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.max_grad_norm
                )

                optimizer.step()
                optimizer.zero_grad()

            # Logging
            epoch_losses['total'] += loss.item()
            for task, task_loss in losses.items():
                if task != 'total':
                    epoch_losses[task] += task_loss.item()

            if batch_idx % config.log_frequency == 0:
                log_metrics(batch_idx, losses)

        # Validation
        if epoch % config.val_frequency == 0:
            val_metrics = validate(model, val_loader, metrics)
            log_validation_metrics(epoch, val_metrics)

        # Checkpointing
        if epoch % config.checkpoint_frequency == 0:
            save_checkpoint(model, optimizer, epoch, val_metrics)

        # Update learning rate
        scheduler.step()
```

### Validation Loop

```python
def validate(model, val_loader, metrics):
    model.eval()
    metrics.reset()

    with torch.no_grad():
        for batch in val_loader:
            # Forward pass
            outputs = model.predict(
                ct_volume=batch['ct_volume'],
                segmentation_mask=batch['segmentation_mask'],
                generate_report=False  # Skip for speed
            )

            # Update metrics
            metrics.update(outputs, batch)

    # Compute final metrics
    final_metrics = metrics.compute()

    return final_metrics
```

## Expected Performance

### Baseline (Zero-shot, before training)

| Task | Metric | Expected Score |
|------|--------|----------------|
| Classification | AUC-ROC | 0.50-0.60 (random-ish) |
| Localization | Dice | 0.20-0.35 (poor) |
| Report Generation | BLEU-4 | 0.05-0.10 (very poor) |

### After Phase 1 (Task-specific pre-training)

| Task | Metric | Expected Score |
|------|--------|----------------|
| Classification | AUC-ROC | 0.75-0.82 |
| | F1 (macro) | 0.65-0.75 |
| Localization | Dice | 0.50-0.65 |
| | IoU | 0.40-0.55 |
| Report Generation | BLEU-4 | 0.15-0.22 |
| | ROUGE-L | 0.30-0.40 |

### After Phase 2 (Joint multi-task training)

| Task | Metric | Expected Score | Improvement |
|------|--------|----------------|-------------|
| Classification | AUC-ROC | 0.85-0.92 | +10-15% |
| | F1 (macro) | 0.75-0.85 | +10-15% |
| Localization | Dice | 0.65-0.80 | +15-25% |
| | IoU | 0.55-0.70 | +15-25% |
| Report Generation | BLEU-4 | 0.25-0.35 | +10-15% |
| | ROUGE-L | 0.40-0.50 | +10-15% |

**Multi-task benefit**: Joint training should improve all tasks due to shared representations and cross-task regularization.

### After Phase 3 (RL fine-tuning for reports)

| Task | Metric | Expected Score | Improvement |
|------|--------|----------------|-------------|
| Report Generation | BLEU-4 | 0.30-0.40 | +5-10% |
| | ROUGE-L | 0.45-0.55 | +5-10% |
| | BERTScore | 0.75-0.85 | +5-10% |
| | Clinical Accuracy | 80-90% | Qualitative |

**RL benefit**: Improved report quality, better adherence to clinical report format, more factually consistent.

### Comparison with Baselines

| Method | AUC (Cls) | Dice (Loc) | BLEU-4 (Gen) |
|--------|-----------|------------|--------------|
| **VLM3D-MultiTask (Ours)** | **0.85-0.92** | **0.65-0.80** | **0.30-0.40** |
| 2D Vision Encoder | 0.78-0.85 | 0.50-0.65 | 0.20-0.28 |
| Task-Specific Models | 0.82-0.88 | 0.60-0.72 | 0.25-0.35 |
| Without Anatomical Context | 0.80-0.86 | 0.55-0.68 | 0.22-0.30 |

**Key advantages**:
- Native 3D processing captures volumetric context
- Anatomical structure encoding improves localization
- Multi-task learning improves all tasks
- LLM-based generation produces coherent reports

## Resource Requirements

### Storage

```
Dataset Storage:
├── CT-RATE (raw): 263 GB
├── CT-RATE (processed NPZ): 180 GB
├── OmniAbnorm: 1 GB
├── Synthetic Reports: 50 MB
├── Validation Set: 12 GB
└── Total Dataset: ~456 GB

Training Artifacts:
├── Checkpoints: 15 GB (6 checkpoints × 2.5 GB)
├── TensorBoard Logs: 5 GB
├── Generated Samples: 2 GB
└── Total Artifacts: ~22 GB

Total Required Storage: ~480 GB
Recommended: 1 TB free space
```

### Memory (GPU VRAM)

```
Model Components:
├── Vision Encoder: ~8 GB
├── Classification Head: ~2 GB
├── Localization Module: ~12 GB
├── Report Generator (Llama-3.1-70B, 4-bit): ~38 GB
└── Total Model: ~60 GB

Training Overhead:
├── Optimizer States: ~8 GB
├── Gradients: ~6 GB
├── Activations (batch): ~10 GB
└── Total Overhead: ~24 GB

Total per GPU: ~84 GB
With Gradient Checkpointing: ~65 GB per GPU

Our Setup (2× H200, 80GB each):
├── GPU 0: Vision encoder + Classification + Localization (~70 GB)
├── GPU 1: Report generator (Llama-3.1-70B) (~75 GB)
└── Model Parallelism: Enabled
```

### Compute Time Breakdown

```
Data Loading: 15% of time
  - I/O from disk: 10%
  - Preprocessing: 5%

Forward Pass: 45% of time
  - Vision encoder: 10%
  - Classification: 2%
  - Localization: 18%
  - Report generation: 15%

Backward Pass: 30% of time
  - Gradient computation: 25%
  - Gradient accumulation: 5%

Optimizer Step: 5% of time

Logging/Checkpointing: 5% of time
```

**Optimization opportunities**:
- Mixed precision (bfloat16): ~30% speedup
- Gradient checkpointing: 40% memory reduction (15% speed cost)
- Efficient data loading: Prefetching, multiple workers
- Compile model with torch.compile: ~20% speedup (PyTorch 2.0+)

## Implementation Files

```
Method/Vision_Encoder_3D/
├── model/
│   ├── vision_encoder.py              # 3D vision encoder (88M)
│   ├── classification_head.py         # Multi-label classifier (1M)
│   ├── localization_module.py         # 3-stage localization (45M)
│   ├── report_generator.py            # LLM-based generator (325M trainable)
│   └── multi_task_model.py            # Unified model wrapper
│
├── training/
│   ├── dataset_multitask.py           # Multi-task dataset loader
│   ├── train_multitask.py             # Training script
│   ├── metrics.py                     # Evaluation metrics
│   └── losses.py                      # Loss functions
│
├── configs/
│   ├── config_large.yaml              # Configuration A (Llama-70B)
│   ├── config_medium.yaml             # Configuration B (Gemma-27B)
│   └── config_small.yaml              # Configuration C (Llama-8B)
│
├── scripts/
│   ├── prepare_data.sh                # Data preparation
│   ├── train_phase1.sh                # Phase 1 training
│   ├── train_phase2.sh                # Phase 2 training
│   ├── train_phase3.sh                # Phase 3 RL training
│   └── evaluate.sh                    # Evaluation script
│
└── docs/
    ├── SUB_OBJECTIVE_3_ARCHITECTURE.md
    ├── SUB_OBJECTIVE_3_TRAINING_PLAN.md  # This file
    └── SUB_OBJECTIVE_3_EVALUATION.md
```

## Quick Start Commands

### 1. Environment Setup

```bash
# Create conda environment
conda create -n vlm3d python=3.10
conda activate vlm3d

# Install PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install dependencies
pip install transformers accelerate peft bitsandbytes
pip install SimpleITK nibabel pillow
pip install tensorboard wandb
pip install nltk rouge bert-score
pip install scikit-learn matplotlib seaborn

# Download NLTK data
python -c "import nltk; nltk.download('punkt')"
```

### 2. Data Preparation

```bash
cd Method/Vision_Encoder_3D

# Prepare CT-RATE data (if not already done)
cd ../../DATA
python prepare_ctrate_for_medsam2.py \
    --volume-dir ./CT-RATE/lung_nodule/volume \
    --mask-dir ./CT-RATE/lung_nodule/masks \
    --output-dir ./CT-RATE/lung_nodule_medsam2 \
    --split all

# Generate synthetic reports
cd ../Method/Vision_Encoder_3D
python scripts/generate_synthetic_reports.py \
    --data-dir ../../DATA/CT-RATE \
    --output-dir ../../DATA/CT-RATE/reports
```

### 3. Training

#### Phase 1: Task-Specific Pre-training

```bash
# Phase 1.1: Classification only
python training/train_multitask.py \
    --config configs/config_large.yaml \
    --phase 1.1 \
    --tasks classification \
    --epochs 10 \
    --batch-size 16 \
    --output-dir ./exp_log_phase1_cls

# Phase 1.2: Localization only
python training/train_multitask.py \
    --config configs/config_large.yaml \
    --phase 1.2 \
    --tasks localization \
    --epochs 15 \
    --batch-size 4 \
    --output-dir ./exp_log_phase1_loc

# Phase 1.3: Report generation alignment
python training/train_multitask.py \
    --config configs/config_large.yaml \
    --phase 1.3 \
    --tasks generation \
    --epochs 5 \
    --batch-size 4 \
    --freeze-vision \
    --output-dir ./exp_log_phase1_gen
```

#### Phase 2: Joint Multi-task Training

```bash
python training/train_multitask.py \
    --config configs/config_large.yaml \
    --phase 2 \
    --tasks classification localization generation \
    --epochs 40 \
    --batch-size 4 \
    --load-phase1-checkpoints \
        ./exp_log_phase1_cls/best_model.pt \
        ./exp_log_phase1_loc/best_model.pt \
        ./exp_log_phase1_gen/best_model.pt \
    --output-dir ./exp_log_phase2_joint
```

#### Phase 3: RL Fine-tuning for Reports

```bash
python training/train_rl_reports.py \
    --config configs/config_large.yaml \
    --phase 3 \
    --method grpo \
    --epochs 10 \
    --batch-size 2 \
    --num-generations 4 \
    --checkpoint ./exp_log_phase2_joint/best_model.pt \
    --output-dir ./exp_log_phase3_rl
```

### 4. Evaluation

```bash
# Evaluate on validation set
python scripts/evaluate.py \
    --checkpoint ./exp_log_phase2_joint/best_model.pt \
    --data-dir ../../DATA/CT-RATE/lung_nodule_medsam2 \
    --split valid \
    --output-dir ./evaluation_results \
    --generate-reports \
    --save-visualizations

# Compute metrics
python scripts/compute_metrics.py \
    --predictions ./evaluation_results/predictions.json \
    --ground-truth ../../DATA/CT-RATE/valid_labels.csv \
    --output ./evaluation_results/metrics.json
```

### 5. Inference

```bash
# Single case inference
python scripts/inference.py \
    --checkpoint ./exp_log_phase2_joint/best_model.pt \
    --input-volume /path/to/ct_volume.nii.gz \
    --input-mask /path/to/seg_mask.nii.gz \
    --output-dir ./inference_output \
    --save-visualizations

# Batch inference
python scripts/batch_inference.py \
    --checkpoint ./exp_log_phase2_joint/best_model.pt \
    --input-dir ../../DATA/CT-RATE/test_volumes \
    --output-dir ./batch_inference_results \
    --batch-size 4
```

## Monitoring Training

### TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir ./exp_log_phase2_joint/tensorboard --port 6006

# View in browser
http://localhost:6006
```

**Metrics to monitor:**
- Training loss (total, per-task)
- Validation metrics (AUC, Dice, BLEU)
- Learning rates
- Gradient norms
- Sample predictions (images, reports)

### Weights & Biases (Optional)

```bash
# Login to W&B
wandb login

# Training will automatically log to W&B
# View at: https://wandb.ai/your-username/vlm3d-multitask
```

### GPU Monitoring

```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# Detailed GPU stats
nvidia-smi dmon -s ucmt -d 1
```

## Troubleshooting

### Out of Memory (OOM)

**Symptoms:** CUDA out of memory error

**Solutions:**
1. Reduce batch size: `--batch-size 2`
2. Increase gradient accumulation: `--gradient-accumulation 16`
3. Enable gradient checkpointing in config
4. Use smaller model configuration (B or C)
5. Reduce number of visual tokens: `num_visual_tokens: 128`

### Slow Training

**Symptoms:** <1 iteration/sec

**Solutions:**
1. Reduce number of data loader workers if CPU-bound
2. Use faster SSD for dataset storage
3. Enable mixed precision (should be default)
4. Use torch.compile for model (PyTorch 2.0+)
5. Profile with PyTorch Profiler to identify bottleneck

### Loss Not Decreasing

**Symptoms:** Loss plateaus or increases

**Solutions:**
1. Lower learning rate by 5-10×
2. Increase warmup steps
3. Check data loading (visualize samples)
4. Verify loss weights are reasonable
5. Try training tasks separately first (Phase 1)
6. Check for NaN gradients: add gradient clipping

### Multi-GPU Issues

**Symptoms:** Training hangs or crashes with multiple GPUs

**Solutions:**
1. Check NCCL backend: `NCCL_DEBUG=INFO`
2. Use different distributed backend: `--backend gloo`
3. Verify network connectivity between GPUs
4. Try single GPU first to isolate issue
5. Update PyTorch and CUDA drivers

## Next Steps

After training is complete:

1. **Evaluation on VLM3D Challenge Test Set**
   - Submit to MICCAI 2025 VLM3D Challenge
   - Compare with other methods on leaderboard

2. **External Validation**
   - Test on other public CT datasets
   - Evaluate generalization to different scanners/protocols

3. **Clinical Validation**
   - Expert radiologist review of generated reports
   - Inter-rater agreement analysis
   - Clinical utility assessment

4. **Model Optimization**
   - Model distillation for faster inference
   - Quantization (INT8/INT4) for deployment
   - ONNX export for clinical systems

5. **Publication**
   - Write paper for MICCAI 2025 or related venues
   - Open-source code and pretrained models
   - Create demo application

## Summary

| Aspect | Details |
|--------|---------|
| **Model** | VLM3D-MultiTask (3D vision encoder + LLM) |
| **Tasks** | Classification (18), Localization (5), Report Generation |
| **Trainable Parameters** | ~460M (out of 70.4B total) |
| **Training Data** | 32K samples (CT-RATE + OmniAbnorm), 276 GB |
| **Training Time** | ~24-28 hours (2× H200 GPUs) |
| **Expected Performance** | AUC 0.85-0.92, Dice 0.65-0.80, BLEU-4 0.30-0.40 |
| **Hardware** | 2× NVIDIA H200 (80GB), 512GB RAM, 1TB SSD |
| **Key Innovation** | Multi-task learning with 3D native vision and anatomical context |

**This training plan provides a complete roadmap from data preparation to final evaluation for the VLM3D multi-task learning system.**
