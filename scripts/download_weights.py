#!/usr/bin/env python3
"""
Download Pre-trained Weights from HuggingFace Hub

Downloads ReCT-VLM model weights for different configurations:
- Large: Llama-3.1-70B (460M trainable parameters)
- Medium: Gemma-2-27B (340M trainable parameters)
- Small: Llama-3.1-8B (280M trainable parameters)

Usage:
    python scripts/download_weights.py --model full --size large
    python scripts/download_weights.py --model vision-encoder
    python scripts/download_weights.py --model all --output-dir ./checkpoints

Author: ReCT-VLM Team
"""

import argparse
import os
from pathlib import Path
from typing import Optional

from huggingface_hub import snapshot_download, hf_hub_download, login
from tqdm import tqdm


# Repository IDs for different model sizes
REPOS = {
    "large": {
        "full": "NEWMES-AI/ReCT-VLM-Large",
        "vision": "NEWMES-AI/ReCT-VisionEncoder-Large",
        "classification": "NEWMES-AI/ReCT-Classifier",
        "localization": "NEWMES-AI/ReCT-Localizer",
        "generation": "NEWMES-AI/ReCT-Generator-Large",
    },
    "medium": {
        "full": "NEWMES-AI/ReCT-VLM-Medium",
        "vision": "NEWMES-AI/ReCT-VisionEncoder-Medium",
        "classification": "NEWMES-AI/ReCT-Classifier",
        "localization": "NEWMES-AI/ReCT-Localizer",
        "generation": "NEWMES-AI/ReCT-Generator-Medium",
    },
    "small": {
        "full": "NEWMES-AI/ReCT-VLM-Small",
        "vision": "NEWMES-AI/ReCT-VisionEncoder-Small",
        "classification": "NEWMES-AI/ReCT-Classifier",
        "localization": "NEWMES-AI/ReCT-Localizer",
        "generation": "NEWMES-AI/ReCT-Generator-Small",
    },
}


def download_model(
    model_name: str,
    size: str = "large",
    output_dir: str = "./checkpoints",
    use_auth_token: Optional[str] = None
):
    """
    Download model from HuggingFace Hub.

    Args:
        model_name: Model component to download ('full', 'vision', 'classification',
                    'localization', 'generation', or 'all')
        size: Model size ('large', 'medium', 'small')
        output_dir: Directory to save downloaded weights
        use_auth_token: HuggingFace authentication token (optional)
    """
    if size not in REPOS:
        print(f"Unknown size: {size}")
        print(f"Available sizes: {list(REPOS.keys())}")
        return

    if model_name == "all":
        # Download all components
        for component in ["full", "vision", "classification", "localization", "generation"]:
            download_model(component, size, output_dir, use_auth_token)
        return

    if model_name not in REPOS[size]:
        print(f"Unknown model: {model_name}")
        print(f"Available models: {list(REPOS[size].keys())}")
        return

    repo_id = REPOS[size][model_name]
    print(f"\n{'='*80}")
    print(f"Downloading {model_name} ({size}) from {repo_id}")
    print(f"{'='*80}")

    # Create output directory
    local_dir = Path(output_dir) / size / model_name
    local_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Download entire repository
        print(f"Saving to: {local_dir}")
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            token=use_auth_token,
            resume_download=True,
        )

        print(f"✓ Successfully downloaded to {local_dir}")

    except Exception as e:
        print(f"✗ Error downloading {model_name}: {e}")
        print(f"  Make sure you have access to the repository: {repo_id}")
        print(f"  You may need to login: huggingface-cli login")


def download_external_models(output_dir: str = "./external_weights"):
    """
    Download external pretrained models (BioBERT, Llama).

    Args:
        output_dir: Directory to save external models
    """
    print(f"\n{'='*80}")
    print("Downloading External Pretrained Models")
    print(f"{'='*80}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # BioBERT
    print("\n1. BioBERT (Bio_ClinicalBERT)")
    biobert_dir = output_path / "biobert"
    biobert_dir.mkdir(exist_ok=True)

    try:
        snapshot_download(
            repo_id="emilyalsentzer/Bio_ClinicalBERT",
            local_dir=biobert_dir,
            local_dir_use_symlinks=False,
        )
        print(f"✓ BioBERT downloaded to {biobert_dir}")
    except Exception as e:
        print(f"✗ Error downloading BioBERT: {e}")

    # Note: Llama models require authentication and agreement to terms
    print("\n2. Llama models require authentication")
    print("   Please download manually from HuggingFace:")
    print("   - https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct")
    print("   - https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct")
    print("   Or use: huggingface-cli download meta-llama/Llama-3.1-70B-Instruct")


def verify_downloads(output_dir: str = "./checkpoints"):
    """
    Verify that downloaded weights are valid.

    Args:
        output_dir: Directory containing downloaded weights
    """
    import torch

    print(f"\n{'='*80}")
    print("Verifying Downloaded Weights")
    print(f"{'='*80}\n")

    output_path = Path(output_dir)

    for size in ["large", "medium", "small"]:
        size_dir = output_path / size
        if not size_dir.exists():
            continue

        print(f"\n{size.upper()} Model:")

        for model_name in ["full", "vision", "classification", "localization"]:
            model_dir = size_dir / model_name
            if not model_dir.exists():
                continue

            # Look for checkpoint files
            checkpoint_files = list(model_dir.glob("*.pt")) + list(model_dir.glob("*.pth"))

            if checkpoint_files:
                checkpoint_path = checkpoint_files[0]
                try:
                    checkpoint = torch.load(checkpoint_path, map_location='cpu')
                    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                    else:
                        state_dict = checkpoint

                    num_params = sum(p.numel() for p in state_dict.values())
                    print(f"  ✓ {model_name}: {checkpoint_path.name} ({num_params/1e6:.1f}M params)")

                except Exception as e:
                    print(f"  ✗ {model_name}: Error loading checkpoint - {e}")
            else:
                print(f"  ⚠ {model_name}: No checkpoint files found")


def main():
    parser = argparse.ArgumentParser(
        description="Download ReCT-VLM pretrained weights from HuggingFace Hub"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["all", "full", "vision", "classification", "localization", "generation"],
        default="full",
        help="Model component to download (default: full)"
    )
    parser.add_argument(
        "--size",
        type=str,
        choices=["large", "medium", "small"],
        default="large",
        help="Model size (default: large)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./checkpoints",
        help="Output directory for downloaded weights (default: ./checkpoints)"
    )
    parser.add_argument(
        "--external",
        action="store_true",
        help="Download external pretrained models (BioBERT, etc.)"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify downloaded weights"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace authentication token (optional)"
    )

    args = parser.parse_args()

    # Login if token provided
    if args.token:
        login(token=args.token)

    print("=" * 80)
    print("ReCT-VLM Weight Downloader")
    print("=" * 80)

    # Download external models
    if args.external:
        download_external_models()

    # Download ReCT-VLM weights
    if not args.verify:
        download_model(
            args.model,
            args.size,
            args.output_dir,
            args.token
        )

    # Verify downloads
    if args.verify:
        verify_downloads(args.output_dir)

    print("\n" + "=" * 80)
    print("Download completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
