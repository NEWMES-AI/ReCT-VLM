#!/usr/bin/env python3
"""
Evaluation Script for ReCT-VLM

Evaluates a trained ReCT-VLM model on test data and computes metrics:
- Classification: AUC, F1, Precision, Recall
- Localization: Dice, IoU
- Report Generation: BLEU, ROUGE, BERTScore

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pt \
                                --config configs/config_large.yaml \
                                --split test

Author: ReCT-VLM Team
"""

import argparse
import os
import sys
from pathlib import Path
import yaml
import json
from typing import Dict, Any
from tqdm import tqdm

import torch
import numpy as np
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.multi_task_model import VLM3DMultiTask
from training.dataset_multitask import CTRATEMultiTaskDataset, collate_fn
from training.metrics import (
    compute_classification_metrics,
    compute_dice_score,
    compute_iou,
    compute_report_metrics
)


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

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params/1e6:.2f}M")

    return model


@torch.no_grad()
def evaluate(
    model: VLM3DMultiTask,
    dataloader: DataLoader,
    device: str,
    save_predictions: bool = False,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluate model on dataset.

    Args:
        model: Model to evaluate
        dataloader: Data loader
        device: Device to use
        save_predictions: Whether to save predictions
        output_dir: Directory to save predictions

    Returns:
        Dictionary of metrics
    """
    model.eval()

    # Collectors
    all_predictions = []
    all_labels = []
    all_masks_pred = []
    all_masks_gt = []
    all_reports_pred = []
    all_reports_gt = []
    all_volume_names = []

    # Evaluate
    for batch in tqdm(dataloader, desc="Evaluating"):
        volumes = batch['volume'].to(device)
        labels = batch['labels'].to(device)
        volume_names = batch['volume_name']

        seg_mask = batch.get('mask', None)
        if seg_mask is not None:
            seg_mask = seg_mask.to(device)

        reports = batch.get('report', None)

        # Forward pass
        outputs = model(
            volumes,
            seg_mask,
            labels=labels,
            reports=reports,
            enable_localization=True,
            enable_generation=(reports is not None)
        )

        # Collect classification predictions
        if 'classification_logits' in outputs:
            predictions = torch.sigmoid(outputs['classification_logits'])
            all_predictions.append(predictions.cpu())
            all_labels.append(labels.cpu())

        # Collect localization predictions
        if 'localization_output' in outputs and seg_mask is not None:
            mask_pred = outputs['localization_output']
            all_masks_pred.append(mask_pred.cpu())
            all_masks_gt.append(seg_mask.cpu())

        # Collect generation predictions
        if 'generated_report' in outputs and reports is not None:
            all_reports_pred.extend(outputs['generated_report'])
            all_reports_gt.extend(reports)

        all_volume_names.extend(volume_names)

    # Compute metrics
    metrics = {}

    # Classification metrics
    if all_predictions:
        print("\nComputing classification metrics...")
        all_predictions = torch.cat(all_predictions).numpy()
        all_labels = torch.cat(all_labels).numpy()

        cls_metrics = compute_classification_metrics(all_predictions, all_labels)
        metrics['classification'] = cls_metrics

        print("Classification Metrics:")
        for k, v in cls_metrics.items():
            print(f"  {k}: {v:.4f}")

    # Localization metrics
    if all_masks_pred:
        print("\nComputing localization metrics...")
        dice_scores = []
        iou_scores = []

        for pred, gt in zip(all_masks_pred, all_masks_gt):
            dice = compute_dice_score(pred, gt)
            iou = compute_iou(pred, gt)
            dice_scores.append(dice)
            iou_scores.append(iou)

        metrics['localization'] = {
            'dice': np.mean(dice_scores),
            'dice_std': np.std(dice_scores),
            'iou': np.mean(iou_scores),
            'iou_std': np.std(iou_scores),
        }

        print("Localization Metrics:")
        print(f"  Dice: {metrics['localization']['dice']:.4f} ± {metrics['localization']['dice_std']:.4f}")
        print(f"  IoU: {metrics['localization']['iou']:.4f} ± {metrics['localization']['iou_std']:.4f}")

    # Report generation metrics
    if all_reports_pred and all_reports_gt:
        print("\nComputing report generation metrics...")
        report_metrics = compute_report_metrics(all_reports_pred, all_reports_gt)
        metrics['generation'] = report_metrics

        print("Report Generation Metrics:")
        for k, v in report_metrics.items():
            print(f"  {k}: {v:.4f}")

    # Save predictions
    if save_predictions and output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        predictions_file = output_path / "predictions.json"

        predictions_data = {
            'volume_names': all_volume_names,
        }

        if all_predictions:
            predictions_data['classification'] = all_predictions.tolist()
            predictions_data['labels'] = all_labels.tolist()

        if all_reports_pred:
            predictions_data['generated_reports'] = all_reports_pred
            predictions_data['reference_reports'] = all_reports_gt

        with open(predictions_file, 'w') as f:
            json.dump(predictions_data, f, indent=2)

        print(f"\nPredictions saved to {predictions_file}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate ReCT-VLM Model")
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
        '--data-dir',
        type=str,
        default=None,
        help='Data directory (overrides config)'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='valid',
        choices=['train', 'valid', 'test'],
        help='Dataset split to evaluate'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='Batch size for evaluation'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of data loader workers'
    )
    parser.add_argument(
        '--save-predictions',
        action='store_true',
        help='Save predictions to file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./evaluation_results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use (cuda or cpu)'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("ReCT-VLM Evaluation")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Config: {args.config}")
    print(f"Split: {args.split}")
    print(f"Device: {args.device}")
    print()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Override data directory if specified
    if args.data_dir:
        config['data']['data_dir'] = args.data_dir

    # Device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load model
    model = load_model(args.checkpoint, config, device)

    # Create dataset
    print(f"\nCreating {args.split} dataset...")

    data_config = config['data']

    if args.split == 'train':
        labels_csv = data_config['train_labels_csv']
        reports_json = data_config.get('train_reports_json', None)
    elif args.split == 'valid':
        labels_csv = data_config['val_labels_csv']
        reports_json = data_config.get('val_reports_json', None)
    else:  # test
        labels_csv = data_config.get('test_labels_csv', data_config['val_labels_csv'])
        reports_json = data_config.get('test_reports_json', None)

    dataset = CTRATEMultiTaskDataset(
        data_dir=data_config['data_dir'],
        labels_csv=labels_csv,
        reports_json=reports_json,
        split=args.split,
        max_slices=data_config['max_slices'],
        target_size=tuple(data_config['target_size']),
        use_augmentation=False,
    )

    print(f"Dataset size: {len(dataset)} samples")

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    print(f"Number of batches: {len(dataloader)}")

    # Evaluate
    print("\n" + "=" * 80)
    print("Starting evaluation...")
    print("=" * 80 + "\n")

    metrics = evaluate(
        model,
        dataloader,
        device,
        save_predictions=args.save_predictions,
        output_dir=args.output_dir
    )

    # Save metrics
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    metrics_file = output_path / f"metrics_{args.split}.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nMetrics saved to {metrics_file}")

    print("\n" + "=" * 80)
    print("Evaluation completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
