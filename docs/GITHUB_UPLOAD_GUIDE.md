# GitHub 업로드 가이드

ReCT-VLM 프로젝트를 GitHub에 업로드하기 위한 상세 가이드입니다.

## 📋 사전 준비 체크리스트

### 1. 필수 파일 확인
- [x] README.md
- [x] LICENSE
- [x] requirements.txt
- [x] setup.py
- [x] .gitignore
- [x] CONTRIBUTING.md

### 2. 코드 구조 확인
```
ReCT-VLM/
├── README.md
├── LICENSE
├── requirements.txt
├── setup.py
├── .gitignore
├── CONTRIBUTING.md
│
├── rect_vlm/                      # 패키지명 변경 필요
│   ├── __init__.py
│   ├── model/
│   ├── training/
│   └── utils/
│
├── configs/
├── scripts/
├── DATA/
├── docs/
├── examples/
└── tests/
```

## 🔧 업로드 전 작업

### Step 1: 디렉토리 구조 재구성

현재 `Method/Vision_Encoder_3D/` 구조를 GitHub repository 루트로 이동해야 합니다.

```bash
# 현재 위치에서
cd /home/work/3D_CT_Foundation_Model/Method/Vision_Encoder_3D/

# 패키지명 변경 (model → rect_vlm)
mv model rect_vlm

# __init__.py 생성
cat > rect_vlm/__init__.py << 'EOF'
"""
ReCT-VLM: Reasoning-Enhanced CT Vision-Language Model
"""

__version__ = "0.1.0"

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
EOF
```

### Step 2: 문서 재배치

```bash
# docs 디렉토리 생성 및 문서 이동
mkdir -p docs/images

# 주요 문서들을 docs로 이동
mv ARCHITECTURE.md docs/
mv TRAINING_PLAN.md docs/TRAINING.md
mv SUB_OBJECTIVE_3_*.md docs/
mv MODULE_TRAINING_DETAILS.md docs/
mv IMPLEMENTATION_SUMMARY.md docs/
mv PROGRESS.md docs/

# README는 루트에 유지
# LICENSE, CONTRIBUTING.md도 루트에 유지
```

### Step 3: 데이터 준비 스크립트 정리

```bash
# DATA 디렉토리 정리
cd DATA/

# README 생성
cat > README.md << 'EOF'
# Dataset Preparation

## CT-RATE Dataset

### Download
\`\`\`bash
python download_dataset.py
\`\`\`

### Prepare for Training
\`\`\`bash
python prepare_ctrate_for_medsam2.py \
    --volume-dir ./CT-RATE/lung_nodule/volume \
    --mask-dir ./CT-RATE/lung_nodule/masks \
    --output-dir ./CT-RATE/lung_nodule_medsam2 \
    --split all
\`\`\`

See [detailed instructions](../docs/DATA_PREPARATION.md).
EOF
```

### Step 4: 가중치 관리

실제 학습된 가중치는 용량이 크므로 HuggingFace Hub에 업로드하고, GitHub에는 다운로드 스크립트만 포함합니다.

```bash
# scripts 디렉토리 생성
mkdir -p scripts

# 가중치 다운로드 스크립트 생성
cat > scripts/download_weights.py << 'EOF'
#!/usr/bin/env python3
"""
Download pre-trained weights from HuggingFace Hub
"""

import argparse
from huggingface_hub import hf_hub_download
import os

REPO_ID = "NEWMES-AI/ReCT-VLM"

WEIGHT_FILES = {
    "vision-encoder": "checkpoints/vision_encoder.pt",
    "classification": "checkpoints/classification_head.pt",
    "localization": "checkpoints/localization_module.pt",
    "full-model": "checkpoints/full_model.pt",
}

def download_weights(component: str, output_dir: str = "./checkpoints"):
    """Download specific component weights."""
    os.makedirs(output_dir, exist_ok=True)

    if component == "all":
        for comp, filename in WEIGHT_FILES.items():
            print(f"Downloading {comp}...")
            hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                local_dir=output_dir
            )
    else:
        filename = WEIGHT_FILES.get(component)
        if filename:
            print(f"Downloading {component}...")
            hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                local_dir=output_dir
            )
        else:
            print(f"Unknown component: {component}")
            print(f"Available: {list(WEIGHT_FILES.keys())}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--component", choices=["all", "vision-encoder", "classification", "localization", "full-model"], default="all")
    parser.add_argument("--output-dir", default="./checkpoints")
    args = parser.parse_args()

    download_weights(args.component, args.output_dir)
EOF

chmod +x scripts/download_weights.py
```

### Step 5: 테스트 파일 생성

```bash
# tests 디렉토리 생성
mkdir -p tests

# 기본 테스트 파일 생성
cat > tests/test_vision_encoder.py << 'EOF'
import pytest
import torch
from rect_vlm.model.vision_encoder import ThreeDVisionEncoder

def test_vision_encoder_forward():
    model = ThreeDVisionEncoder(
        in_channels=1,
        embed_dim=768,
        depth=12,
        num_heads=12
    )

    x = torch.randn(1, 1, 64, 512, 512)
    seg_mask = torch.randint(0, 20, (1, 64, 512, 512))

    outputs = model(x, seg_mask)

    assert outputs['global_features'].shape == (1, 768)
    assert outputs['local_features'].shape[0] == 1
    assert outputs['region_features'].shape == (1, 20, 768)

if __name__ == "__main__":
    test_vision_encoder_forward()
    print("✓ Vision encoder test passed")
EOF

# __init__.py
touch tests/__init__.py
```

## 📤 GitHub 업로드 절차

### Step 1: Git 초기화

```bash
# 프로젝트 루트로 이동
cd /home/work/3D_CT_Foundation_Model/Method/Vision_Encoder_3D/

# Git 초기화
git init

# .gitignore 적용
git add .gitignore
git commit -m "chore: add .gitignore"
```

### Step 2: 초기 커밋

```bash
# 모든 파일 추가
git add .

# 커밋 (대용량 파일 제외됨)
git commit -m "Initial commit: ReCT-VLM implementation

- Add 3D Vision Encoder with slice-aware attention
- Add multi-label classification head with BioBERT
- Add 3-stage lesion localization module
- Add LLM-based report generator with LoRA
- Add multi-task learning integration
- Add training infrastructure and metrics
- Add comprehensive documentation"
```

### Step 3: Remote Repository 연결

```bash
# GitHub repository 연결
git remote add origin https://github.com/NEWMES-AI/ReCT-VLM.git

# 브랜치 확인
git branch -M main
```

### Step 4: Push to GitHub

```bash
# 첫 번째 push
git push -u origin main
```

## 🔐 대용량 파일 처리 (HuggingFace)

### 가중치 업로드 (학습 완료 후)

```bash
# HuggingFace Hub 설치
pip install huggingface_hub

# 로그인
huggingface-cli login

# Repository 생성 (웹에서 먼저 생성 권장)
# https://huggingface.co/new

# 가중치 업로드
python << 'EOF'
from huggingface_hub import HfApi

api = HfApi()

# Upload checkpoints
api.upload_file(
    path_or_fileobj="./checkpoints/full_model.pt",
    path_in_repo="checkpoints/full_model.pt",
    repo_id="NEWMES-AI/ReCT-VLM",
    repo_type="model"
)

# Upload configuration
api.upload_file(
    path_or_fileobj="./configs/config_large.yaml",
    path_in_repo="configs/config_large.yaml",
    repo_id="NEWMES-AI/ReCT-VLM",
    repo_type="model"
)
EOF
```

## 📊 데이터셋 처리

### CT-RATE 데이터셋

CT-RATE는 HuggingFace Datasets에 이미 호스팅되어 있으므로, README에 다운로드 방법만 명시:

```markdown
## Dataset

We use the [CT-RATE dataset](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE).

Download instructions in [DATA/README.md](DATA/README.md).
```

### OmniAbnorm CoT 데이터셋

생성된 CoT 데이터를 HuggingFace Datasets로 공유:

```bash
# dataset_dict 생성 및 업로드
python << 'EOF'
from datasets import Dataset, DatasetDict
import json

# Load generated CoT data
with open("DATA/OmniAbnorm/cot_data.json") as f:
    data = json.load(f)

# Create dataset
dataset = Dataset.from_dict(data)

# Upload to HuggingFace
dataset.push_to_hub("NEWMES-AI/OmniAbnorm-CoT")
EOF
```

## ✅ 업로드 후 확인 사항

### GitHub에서 확인

1. [ ] README.md가 올바르게 표시되는지
2. [ ] 라이선스가 자동 인식되는지
3. [ ] .gitignore가 작동하는지 (대용량 파일 제외)
4. [ ] 모든 문서가 정상적으로 표시되는지
5. [ ] 이미지 링크가 작동하는지

### 설정 확인

1. [ ] Repository 설정
   - Description 추가
   - Topics 추가 (medical-imaging, computer-vision, vision-language-model, pytorch, transformers)
   - Website 추가 (있다면)

2. [ ] GitHub Actions 설정 (Optional)
   - CI/CD 파이프라인
   - 자동 테스트
   - 문서 빌드

3. [ ] Issues 템플릿 생성
   - Bug report
   - Feature request

## 📝 추가 작업 (Optional)

### GitHub Pages 설정

```bash
# gh-pages 브랜치 생성
git checkout --orphan gh-pages
git rm -rf .

# 문서 사이트 생성 (Sphinx or MkDocs)
# ...

git add .
git commit -m "docs: initialize GitHub Pages"
git push origin gh-pages
```

### Releases 생성

```bash
# Tag 생성
git tag -a v0.1.0 -m "Initial release: ReCT-VLM v0.1.0"
git push origin v0.1.0

# GitHub에서 Release 생성
# - Release notes 작성
# - Pre-trained weights 링크 추가
# - Changelog 포함
```

### Badges 추가

README.md 상단에 추가할 badges:
- Paper link (arXiv)
- License
- Python version
- PyTorch version
- CI status
- Code coverage
- Downloads

## 🚨 주의사항

### 절대 업로드하면 안 되는 것

1. **대용량 파일**
   - 모델 체크포인트 (*.pt, *.pth)
   - 데이터셋 파일 (*.nii.gz, *.npz)
   - 실험 결과 파일

2. **민감 정보**
   - API keys
   - HuggingFace tokens
   - 개인 정보

3. **임시 파일**
   - 캐시 파일
   - 로그 파일
   - __pycache__

### Git LFS 사용 (Optional)

대용량 파일을 GitHub에 올려야 한다면 Git LFS 사용:

```bash
# Git LFS 설치
git lfs install

# 추적할 파일 타입 지정
git lfs track "*.pt"
git lfs track "*.pth"

# .gitattributes 추가
git add .gitattributes
git commit -m "chore: configure Git LFS"
```

하지만 **HuggingFace Hub 사용을 권장**합니다.

## 📮 최종 체크리스트

업로드 전 최종 확인:

- [ ] 모든 코드가 정상 작동하는지 테스트
- [ ] README.md가 완성되었는지
- [ ] LICENSE가 포함되었는지
- [ ] .gitignore가 올바르게 설정되었는지
- [ ] 민감 정보가 제거되었는지
- [ ] 문서가 완성되었는지
- [ ] 예제 코드가 작동하는지
- [ ] 설치 가이드가 정확한지

## 🎉 완료!

모든 절차를 완료했다면:

1. GitHub repository: https://github.com/NEWMES-AI/ReCT-VLM
2. HuggingFace models: https://huggingface.co/NEWMES-AI
3. Documentation: GitHub Pages or ReadTheDocs

**다음 단계**:
- CI/CD 설정
- 문서 사이트 구축
- 커뮤니티 관리
- Issue 대응
- 지속적 업데이트
