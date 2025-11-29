#!/usr/bin/env python3
"""
Inference Script for ReCT-VLM

Run inference on CT volumes to generate:
- Multi-label disease classification
- Lesion localization masks
- Radiology reports

Usage:
    # Single volume
    python scripts/inference.py --checkpoint checkpoints/best_model.pt \
                                 --config configs/config_large.yaml \
                                 --input data/ct_volume.nii.gz \
                                 --mask data/ct_mask.nii.gz \
                                 --output results/

    # Directory of volumes
    python scripts/inference.py --checkpoint checkpoints/best_model.pt \
                                 --config configs/config_large.yaml \
                                 --input-dir data/volumes/ \
                                 --output-dir results/

Author: ReCT-VLM Team
"""

import argparse
import os
import sys
from pathlib import Path
import yaml
import json
from typing import Dict, Any, Optional, List
from tqdm import tqdm

import torch
import torch.nn.functional as F
import numpy as np
import nibabel as nib

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.multi_task_model import VLM3DMultiTask


DISEASE_NAMES = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration',
    'Mass', 'Nodule', 'Pleural_Thickening', 'Pneumonia',
    'Pneumothorax', 'Lung nodule', 'Lung opacity', 'Enlarged lymph nodes',
    'Pleural effusion', 'Pericardial effusion'
]


def load_model(checkpoint_path: str, config: Dict[str, Any], device: str) -> VLM3DMultiTask:
    """Load model from checkpoint."""
    print(f"Loading model from {checkpoint_path}")

    model_config = config['model']

    model = VLM3DMultiTask(
        in_channels=model_config['in_channels'],
        patch_size=tuple(model_config['patch_size']),
        embed_dim=model_config['embed_dim'],
        depth=model_config['depth'],
        num_heads=model_config['num_heads'],
        mlp_ratio=model_config['mlp_ratio'],
        num_regions=model_config['num_regions'],
        num_classes=model_config['num_classes'],
        biobert_model=model_config['biobert_model'],
        localization_diseases=model_config['localization_diseases'],
        llm_model_name=model_config.get('llm_model_name', 'meta-llama/Llama-3.1-70B-Instruct'),
        enable_classification=True,
        enable_localization=True,
        enable_generation=True,
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Handle DDP wrapped models
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    print("Model loaded successfully")

    return model


def load_volume(path: str) -> np.ndarray:
    """Load CT volume from file."""
    if path.endswith('.npz'):
        data = np.load(path)
        volume = data['volume'] if 'volume' in data else data['imgs']
    else:
        nii = nib.load(path)
        volume = nii.get_fdata()

    return volume.astype(np.float32)


def load_mask(path: str) -> Optional[np.ndarray]:
    """Load segmentation mask from file."""
    if not os.path.exists(path):
        return None

    if path.endswith('.npz'):
        data = np.load(path)
        mask = data['mask'] if 'mask' in data else data['gts']
    else:
        nii = nib.load(path)
        mask = nii.get_fdata()

    return mask.astype(np.uint8)


def preprocess_volume(
    volume: np.ndarray,
    max_slices: int = 64,
    target_size: tuple = (512, 512)
) -> torch.Tensor:
    """Preprocess CT volume for model input."""
    # CT windowing (lung window)
    window_center = 40
    window_width = 400
    lower = window_center - window_width // 2
    upper = window_center + window_width // 2

    volume = np.clip(volume, lower, upper)
    volume = (volume - lower) / (upper - lower) * 255.0

    # Sample slices if needed
    D, H, W = volume.shape
    if D > max_slices:
        indices = np.linspace(0, D - 1, max_slices).astype(int)
        volume = volume[indices]

    # Convert to tensor and resize
    volume_tensor = torch.from_numpy(volume).float()
    volume_tensor = volume_tensor.unsqueeze(1)  # (D, 1, H, W)

    volume_resized = F.interpolate(
        volume_tensor,
        size=target_size,
        mode='bilinear',
        align_corners=False
    ).squeeze(1)  # (D, H', W')

    # Normalize to [0, 1]
    volume_resized = volume_resized / 255.0

    # Add batch and channel dimensions: (1, 1, D, H, W)
    volume_resized = volume_resized.unsqueeze(0).unsqueeze(0)

    return volume_resized


def preprocess_mask(
    mask: Optional[np.ndarray],
    max_slices: int = 64,
    target_size: tuple = (512, 512)
) -> Optional[torch.Tensor]:
    """Preprocess segmentation mask."""
    if mask is None:
        return None

    # Sample slices if needed
    D, H, W = mask.shape
    if D > max_slices:
        indices = np.linspace(0, D - 1, max_slices).astype(int)
        mask = mask[indices]

    # Convert to tensor and resize
    mask_tensor = torch.from_numpy(mask).float()
    mask_tensor = mask_tensor.unsqueeze(1)  # (D, 1, H, W)

    mask_resized = F.interpolate(
        mask_tensor,
        size=target_size,
        mode='nearest'
    ).squeeze(1).long()  # (D, H', W')

    # Add batch dimension: (1, D, H, W)
    mask_resized = mask_resized.unsqueeze(0)

    return mask_resized


@torch.no_grad()
def inference(
    model: VLM3DMultiTask,
    volume: torch.Tensor,
    mask: Optional[torch.Tensor],
    device: str,
    generate_report: bool = True
) -> Dict[str, Any]:
    """
    Run inference on a single volume.

    Args:
        model: Trained model
        volume: Preprocessed CT volume (1, 1, D, H, W)
        mask: Segmentation mask (1, D, H, W) or None
        device: Device to use
        generate_report: Whether to generate radiology report

    Returns:
        Dictionary containing predictions
    """
    model.eval()

    volume = volume.to(device)
    if mask is not None:
        mask = mask.to(device)

    # Forward pass
    outputs = model(
        volume,
        mask,
        enable_localization=True,
        enable_generation=generate_report
    )

    results = {}

    # Classification predictions
    if 'classification_logits' in outputs:
        logits = outputs['classification_logits']
        probs = torch.sigmoid(logits).cpu().numpy()[0]  # (18,)

        # Create disease predictions
        predictions = []
        for disease, prob in zip(DISEASE_NAMES, probs):
            predictions.append({
                'disease': disease,
                'probability': float(prob),
                'predicted': bool(prob > 0.5)
            })

        results['classification'] = predictions

    # Localization predictions
    if 'localization_output' in outputs:
        loc_mask = outputs['localization_output'].cpu().numpy()[0]  # (D, H, W)
        results['localization_mask'] = loc_mask

    # Generated report
    if 'generated_report' in outputs:
        report = outputs['generated_report'][0] if isinstance(outputs['generated_report'], list) else outputs['generated_report']
        results['report'] = report

    return results


def save_results(
    results: Dict[str, Any],
    output_dir: str,
    volume_name: str,
    save_mask: bool = True
):
    """Save inference results."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save classification results
    if 'classification' in results:
        cls_file = output_path / f"{volume_name}_classification.json"
        with open(cls_file, 'w') as f:
            json.dump(results['classification'], f, indent=2)

    # Save localization mask
    if save_mask and 'localization_mask' in results:
        mask = results['localization_mask']
        mask_nii = nib.Nifti1Image(mask, affine=np.eye(4))
        mask_file = output_path / f"{volume_name}_localization.nii.gz"
        nib.save(mask_nii, mask_file)

    # Save report
    if 'report' in results:
        report_file = output_path / f"{volume_name}_report.txt"
        with open(report_file, 'w') as f:
            f.write(results['report'])

    # Save summary
    summary = {
        'volume_name': volume_name,
        'classification': results.get('classification', []),
        'report': results.get('report', ''),
        'has_localization': 'localization_mask' in results,
    }

    summary_file = output_path / f"{volume_name}_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="ReCT-VLM Inference")
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config file'
    )
    parser.add_argument(
        '--input',
        type=str,
        help='Input CT volume file (.nii.gz or .npz)'
    )
    parser.add_argument(
        '--mask',
        type=str,
        default=None,
        help='Input segmentation mask file (optional)'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        help='Directory containing multiple volumes'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./inference_results',
        help='Output directory'
    )
    parser.add_argument(
        '--no-report',
        action='store_true',
        help='Disable report generation'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use (cuda or cpu)'
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.input and not args.input_dir:
        parser.error("Either --input or --input-dir must be specified")

    print("=" * 80)
    print("ReCT-VLM Inference")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Config: {args.config}")
    print(f"Device: {args.device}")
    print()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load model
    model = load_model(args.checkpoint, config, device)

    # Get list of volumes to process
    if args.input:
        volumes = [(args.input, args.mask, Path(args.input).stem)]
    else:
        # Find all volumes in directory
        input_path = Path(args.input_dir)
        volume_files = list(input_path.glob("*.nii.gz")) + list(input_path.glob("*.npz"))
        volumes = [(str(f), None, f.stem) for f in volume_files]

    print(f"\nProcessing {len(volumes)} volume(s)...\n")

    # Process each volume
    for volume_path, mask_path, volume_name in tqdm(volumes):
        try:
            # Load and preprocess
            volume = load_volume(volume_path)
            mask = load_mask(mask_path) if mask_path else None

            volume_tensor = preprocess_volume(
                volume,
                max_slices=config['data']['max_slices'],
                target_size=tuple(config['data']['target_size'])
            )

            mask_tensor = None
            if mask is not None:
                mask_tensor = preprocess_mask(
                    mask,
                    max_slices=config['data']['max_slices'],
                    target_size=tuple(config['data']['target_size'])
                )

            # Run inference
            results = inference(
                model,
                volume_tensor,
                mask_tensor,
                device,
                generate_report=(not args.no_report)
            )

            # Save results
            save_results(results, args.output, volume_name)

            # Print summary
            print(f"\n{volume_name}:")
            if 'classification' in results:
                positive = [p for p in results['classification'] if p['predicted']]
                print(f"  Detected diseases: {len(positive)}")
                for p in positive[:5]:  # Show top 5
                    print(f"    - {p['disease']}: {p['probability']:.3f}")

            if 'report' in results:
                print(f"  Report: {results['report'][:100]}...")

        except Exception as e:
            print(f"Error processing {volume_name}: {e}")
            continue

    print("\n" + "=" * 80)
    print("Inference completed!")
    print(f"Results saved to {args.output}")
    print("=" * 80)


if __name__ == "__main__":
    main()
