# 가중치 관리 가이드

ReCT-VLM 모델 가중치의 저장, 공유, 다운로드 방법을 설명합니다.

## 📦 가중치 구조

### 학습된 가중치 파일들

```
checkpoints/
├── vision_encoder/
│   ├── model_epoch_50.pt              # Vision encoder (88M)
│   ├── best_model.pt                  # Best checkpoint
│   └── config.yaml                    # Training config
│
├── classification/
│   ├── model_epoch_10.pt              # Classification head (1M)
│   ├── best_model.pt
│   └── biobert_projectors.pt          # Vision/Text projectors
│
├── localization/
│   ├── model_epoch_15.pt              # Localization module (45M)
│   ├── best_model.pt
│   ├── text_embedder.pt               # BioBERT embeddings
│   ├── denoising_transformer.pt       # Denoising module
│   └── attention_unet.pt              # U-Net module
│
├── report_generator/
│   ├── projector_epoch_5.pt           # Vision-to-LLM projector (256M)
│   ├── lora_adapters_epoch_5/         # LoRA weights (325M)
│   │   ├── adapter_config.json
│   │   └── adapter_model.bin
│   └── best_projector.pt
│
└── full_model/
    ├── model_phase2_epoch_40.pt       # Complete multi-task model
    ├── model_phase3_epoch_10.pt       # After RL fine-tuning
    └── best_model.pt                  # Best overall model

Total size: ~2-3 GB (without LLM base weights)
```

### 외부 Pre-trained Weights

다운로드가 필요한 외부 모델들:

```
external_weights/
├── biobert/
│   └── Bio_ClinicalBERT/              # 110M, ~440 MB
│       ├── pytorch_model.bin
│       ├── config.json
│       └── vocab.txt
│
└── llama/
    └── Llama-3.1-70B-Instruct/        # 70B, ~38 GB (4-bit)
        ├── model-*.safetensors
        ├── config.json
        ├── tokenizer.json
        └── tokenizer_config.json
```

## 🎯 HuggingFace Hub 업로드

### 1. Repository 생성

HuggingFace에 model repository 생성:

```bash
# HuggingFace CLI 로그인
huggingface-cli login

# Or Python API
from huggingface_hub import HfApi
api = HfApi()

# Create repository
api.create_repo(
    repo_id="NEWMES-AI/ReCT-VLM-Large",
    repo_type="model",
    private=False
)
```

### 2. 가중치 업로드

#### 전체 모델 업로드

```python
from huggingface_hub import HfApi
import os

api = HfApi()
repo_id = "NEWMES-AI/ReCT-VLM-Large"

# Upload full model checkpoint
api.upload_file(
    path_or_fileobj="checkpoints/full_model/best_model.pt",
    path_in_repo="pytorch_model.bin",
    repo_id=repo_id,
)

# Upload config
api.upload_file(
    path_or_fileobj="configs/config_large.yaml",
    path_in_repo="config.yaml",
    repo_id=repo_id,
)

# Upload README
api.upload_file(
    path_or_fileobj="README.md",
    path_in_repo="README.md",
    repo_id=repo_id,
)
```

#### 개별 컴포넌트 업로드

```python
# Vision Encoder
api.create_repo("NEWMES-AI/ReCT-VisionEncoder", repo_type="model")
api.upload_file(
    path_or_fileobj="checkpoints/vision_encoder/best_model.pt",
    path_in_repo="pytorch_model.bin",
    repo_id="NEWMES-AI/ReCT-VisionEncoder",
)

# Classification Head
api.create_repo("NEWMES-AI/ReCT-Classifier", repo_type="model")
api.upload_file(
    path_or_fileobj="checkpoints/classification/best_model.pt",
    path_in_repo="pytorch_model.bin",
    repo_id="NEWMES-AI/ReCT-Classifier",
)

# Localization Module
api.create_repo("NEWMES-AI/ReCT-Localizer", repo_type="model")
api.upload_folder(
    folder_path="checkpoints/localization/",
    repo_id="NEWMES-AI/ReCT-Localizer",
)
```

### 3. Model Card 작성

HuggingFace에 업로드할 `README.md` (Model Card):

```markdown
---
license: apache-2.0
tags:
- medical-imaging
- vision-language
- multi-task-learning
- ct-scan
- radiology
datasets:
- ibrahimhamamci/CT-RATE
language:
- en
metrics:
- accuracy
- f1
- dice
- bleu
library_name: pytorch
---

# ReCT-VLM: Reasoning-Enhanced CT Vision-Language Model

ReCT-VLM is a comprehensive multi-task learning framework for CT image analysis.

## Model Details

- **Model Type**: Multi-task Vision-Language Model
- **Architecture**: 3D Vision Encoder + Multi-label Classifier + Lesion Localizer + Report Generator
- **Parameters**: 70.5B total (460M trainable)
- **Training Data**: CT-RATE dataset (21,907 cases)

## Usage

\`\`\`python
from rect_vlm import ReCTVLM

model = ReCTVLM.from_pretrained("NEWMES-AI/ReCT-VLM-Large")
predictions = model.predict(ct_volume, segmentation_mask)
\`\`\`

## Citation

\`\`\`bibtex
@article{rect-vlm2025,
  title={ReCT-VLM: Reasoning-Enhanced CT Vision-Language Model},
  author={},
  year={2025}
}
\`\`\`
```

## 📥 가중치 다운로드

### 사용자를 위한 다운로드 스크립트

`scripts/download_weights.py`:

```python
#!/usr/bin/env python3
"""Download pre-trained weights from HuggingFace Hub"""

import argparse
from huggingface_hub import snapshot_download, hf_hub_download
import os

# Repository IDs
REPOS = {
    "full": "NEWMES-AI/ReCT-VLM-Large",
    "medium": "NEWMES-AI/ReCT-VLM-Medium",
    "small": "NEWMES-AI/ReCT-VLM-Small",
    "vision": "NEWMES-AI/ReCT-VisionEncoder",
    "classification": "NEWMES-AI/ReCT-Classifier",
    "localization": "NEWMES-AI/ReCT-Localizer",
}

def download_model(model_name: str, output_dir: str = "./checkpoints"):
    """Download model from HuggingFace Hub."""

    if model_name not in REPOS:
        print(f"Unknown model: {model_name}")
        print(f"Available: {list(REPOS.keys())}")
        return

    repo_id = REPOS[model_name]
    print(f"Downloading {model_name} from {repo_id}...")

    # Download entire repository
    local_dir = os.path.join(output_dir, model_name)
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False
    )

    print(f"✓ Downloaded to {local_dir}")

def download_all(output_dir: str = "./checkpoints"):
    """Download all available models."""
    for model_name in REPOS.keys():
        download_model(model_name, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download ReCT-VLM weights")
    parser.add_argument(
        "--model",
        choices=["all"] + list(REPOS.keys()),
        default="full",
        help="Model to download"
    )
    parser.add_argument(
        "--output-dir",
        default="./checkpoints",
        help="Output directory"
    )

    args = parser.parse_args()

    if args.model == "all":
        download_all(args.output_dir)
    else:
        download_model(args.model, args.output_dir)
```

### 사용 예제

```bash
# 전체 모델 다운로드 (Large)
python scripts/download_weights.py --model full

# 특정 컴포넌트만 다운로드
python scripts/download_weights.py --model vision
python scripts/download_weights.py --model classification

# Medium 모델
python scripts/download_weights.py --model medium

# 모든 모델
python scripts/download_weights.py --model all
```

## 🔄 버전 관리

### 가중치 버저닝

```
ReCT-VLM-Large/
├── v0.1.0/                 # Initial release
│   └── pytorch_model.bin
├── v0.2.0/                 # Improved after RL
│   └── pytorch_model.bin
└── main/                   # Latest stable
    └── pytorch_model.bin
```

### Git Tags 사용

```bash
# Tag checkpoint
git tag -a weights-v0.1.0 -m "Release v0.1.0 weights"
git push origin weights-v0.1.0

# Download specific version
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="NEWMES-AI/ReCT-VLM-Large",
    filename="pytorch_model.bin",
    revision="v0.1.0"  # Specific version
)
```

## 💾 체크포인트 저장 형식

### PyTorch 형식

```python
# 저장
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'best_metric': best_metric,
    'config': config,
}
torch.save(checkpoint, 'checkpoint.pt')

# 로드
checkpoint = torch.load('checkpoint.pt')
model.load_state_dict(checkpoint['model_state_dict'])
```

### SafeTensors 형식 (권장)

```python
from safetensors.torch import save_file, load_file

# 저장
save_file(model.state_dict(), 'model.safetensors')

# 로드
state_dict = load_file('model.safetensors')
model.load_state_dict(state_dict)
```

SafeTensors의 장점:
- 더 안전 (임의 코드 실행 불가)
- 더 빠른 로딩
- 메모리 효율적
- 프레임워크 간 호환성

## 🌐 공개 vs 비공개

### 공개 (Public)

**장점**:
- 연구 재현성
- 커뮤니티 기여
- Citation 증가

**권장 시기**:
- 논문 accept 후
- 충분한 검증 완료

### 비공개 (Private)

**사용 시기**:
- 논문 제출 중
- 추가 실험 진행 중
- 상업적 사용 고려

```python
# Private repository 생성
api.create_repo(
    repo_id="NEWMES-AI/ReCT-VLM-Private",
    repo_type="model",
    private=True  # Private
)
```

## 📊 가중치 크기 최적화

### 양자화 (Quantization)

```python
# 4-bit quantization
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# 70B → ~35 GB
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-70B-Instruct",
    quantization_config=quantization_config
)
```

### 모델 프루닝

```python
import torch.nn.utils.prune as prune

# Prune 20% of weights
prune.l1_unstructured(module, name="weight", amount=0.2)
```

## 🔐 접근 제어

### Gated Model

논문 accept 전까지 gated access:

```python
# HuggingFace Model Card에 추가
---
extra_gated_prompt: "Request access to ReCT-VLM weights"
extra_gated_fields:
  Name: text
  Email: text
  Organization: text
  Purpose: text
---
```

### API Token

```bash
# User needs HF token to download
huggingface-cli login

# In code
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="NEWMES-AI/ReCT-VLM-Large",
    filename="pytorch_model.bin",
    use_auth_token=True  # Requires login
)
```

## 📝 체크리스트

업로드 전 확인:

- [ ] 모델이 정상 작동하는지 테스트
- [ ] 민감 정보 제거 (paths, tokens, etc.)
- [ ] Model Card 작성 완료
- [ ] LICENSE 포함
- [ ] 사용 예제 포함
- [ ] 버전 태그 지정
- [ ] 파일 크기 최적화

다운로드 스크립트 테스트:

- [ ] 다운로드 스크립트 작동 확인
- [ ] 로드 후 inference 테스트
- [ ] 다양한 환경에서 테스트

## 🎉 완료

가중치가 성공적으로 업로드되면:

1. **HuggingFace Hub**: https://huggingface.co/NEWMES-AI
2. **Model Zoo**: README.md에 링크 추가
3. **Documentation**: 다운로드 방법 문서화
4. **Announcement**: 커뮤니티에 공지

---

**문의**: GitHub Issues 또는 이메일
