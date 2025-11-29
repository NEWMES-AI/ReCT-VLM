"""
Multi-task Trainer for ReCT-VLM

Handles training loop for all three phases:
- Phase 1: Vision Encoder + Classification
- Phase 2: Multi-task (Classification + Localization)
- Phase 3: Full system with Report Generation

Author: ReCT-VLM Team
"""

import os
import time
from typing import Dict, Optional, Any
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import wandb

from .metrics import (
    compute_classification_metrics,
    compute_dice_score,
    compute_iou,
    compute_report_metrics
)


class MultiTaskTrainer:
    """
    Trainer for multi-task ReCT-VLM model.

    Args:
        model: Multi-task model
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to train on
        output_dir: Directory to save checkpoints and logs
        phase: Training phase (1, 2, or 3)
        max_epochs: Maximum number of epochs
        mixed_precision: Whether to use mixed precision training
        gradient_clip: Gradient clipping value
        log_interval: Log every N batches
        eval_interval: Evaluate every N epochs
        save_interval: Save checkpoint every N epochs
        use_wandb: Whether to use Weights & Biases logging
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        device: str = "cuda",
        output_dir: str = "./outputs",
        phase: int = 1,
        max_epochs: int = 50,
        mixed_precision: bool = True,
        gradient_clip: float = 1.0,
        log_interval: int = 10,
        eval_interval: int = 1,
        save_interval: int = 5,
        use_wandb: bool = False,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_dir = Path(output_dir)
        self.phase = phase
        self.max_epochs = max_epochs
        self.mixed_precision = mixed_precision
        self.gradient_clip = gradient_clip
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.use_wandb = use_wandb

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)

        # Mixed precision scaler
        self.scaler = GradScaler() if mixed_precision else None

        # Tracking
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = 0.0

        # Move model to device
        self.model.to(device)

        print(f"Trainer initialized for Phase {phase}")
        print(f"Output directory: {self.output_dir}")
        print(f"Device: {device}")
        print(f"Mixed precision: {mixed_precision}")

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = {
            'total': 0.0,
            'classification': 0.0,
            'localization': 0.0,
            'generation': 0.0,
        }

        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch}/{self.max_epochs}"
        )

        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            volumes = batch['volume'].to(self.device)  # (B, 1, D, H, W)
            labels = batch['labels'].to(self.device)  # (B, 18)

            seg_mask = batch.get('mask', None)
            if seg_mask is not None:
                seg_mask = seg_mask.to(self.device)

            reports = batch.get('report', None)

            # Forward pass
            self.optimizer.zero_grad()

            with autocast(enabled=self.mixed_precision):
                if self.phase == 1:
                    # Phase 1: Vision Encoder + Classification
                    outputs = self.model(volumes, seg_mask)
                    loss = outputs['classification_loss']

                elif self.phase == 2:
                    # Phase 2: Multi-task (Classification + Localization)
                    outputs = self.model(
                        volumes,
                        seg_mask,
                        labels=labels,
                        enable_localization=True
                    )
                    loss = outputs['total_loss']

                elif self.phase == 3:
                    # Phase 3: Full system with Report Generation
                    outputs = self.model(
                        volumes,
                        seg_mask,
                        labels=labels,
                        reports=reports,
                        enable_localization=True,
                        enable_generation=True
                    )
                    loss = outputs['total_loss']

            # Backward pass
            if self.mixed_precision:
                self.scaler.scale(loss).backward()
                if self.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip
                    )
                self.optimizer.step()

            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()

            # Accumulate losses
            epoch_losses['total'] += loss.item()
            if 'classification_loss' in outputs:
                epoch_losses['classification'] += outputs['classification_loss'].item()
            if 'localization_loss' in outputs:
                epoch_losses['localization'] += outputs['localization_loss'].item()
            if 'generation_loss' in outputs:
                epoch_losses['generation'] += outputs['generation_loss'].item()

            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item(),
                'lr': self.optimizer.param_groups[0]['lr']
            })

            # Log to wandb
            if self.use_wandb and batch_idx % self.log_interval == 0:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/lr': self.optimizer.param_groups[0]['lr'],
                    'global_step': self.global_step
                })

            self.global_step += 1

        # Average losses
        num_batches = len(self.train_loader)
        for key in epoch_losses:
            epoch_losses[key] /= num_batches

        return epoch_losses

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()

        all_predictions = []
        all_labels = []
        all_masks_pred = []
        all_masks_gt = []
        all_reports_pred = []
        all_reports_gt = []

        val_losses = {
            'total': 0.0,
            'classification': 0.0,
            'localization': 0.0,
            'generation': 0.0,
        }

        for batch in tqdm(self.val_loader, desc="Validation"):
            volumes = batch['volume'].to(self.device)
            labels = batch['labels'].to(self.device)

            seg_mask = batch.get('mask', None)
            if seg_mask is not None:
                seg_mask = seg_mask.to(self.device)

            reports = batch.get('report', None)

            # Forward pass
            with autocast(enabled=self.mixed_precision):
                if self.phase == 1:
                    outputs = self.model(volumes, seg_mask)
                elif self.phase == 2:
                    outputs = self.model(
                        volumes, seg_mask, labels=labels,
                        enable_localization=True
                    )
                elif self.phase == 3:
                    outputs = self.model(
                        volumes, seg_mask, labels=labels,
                        reports=reports,
                        enable_localization=True,
                        enable_generation=True
                    )

            # Accumulate losses
            if 'total_loss' in outputs:
                val_losses['total'] += outputs['total_loss'].item()
            if 'classification_loss' in outputs:
                val_losses['classification'] += outputs['classification_loss'].item()
            if 'localization_loss' in outputs:
                val_losses['localization'] += outputs['localization_loss'].item()
            if 'generation_loss' in outputs:
                val_losses['generation'] += outputs['generation_loss'].item()

            # Collect predictions for metrics
            if 'classification_logits' in outputs:
                predictions = torch.sigmoid(outputs['classification_logits'])
                all_predictions.append(predictions.cpu())
                all_labels.append(labels.cpu())

            if 'localization_output' in outputs and seg_mask is not None:
                mask_pred = outputs['localization_output']
                all_masks_pred.append(mask_pred.cpu())
                all_masks_gt.append(seg_mask.cpu())

            if 'generated_report' in outputs and reports is not None:
                all_reports_pred.extend(outputs['generated_report'])
                all_reports_gt.extend(reports)

        # Average losses
        num_batches = len(self.val_loader)
        for key in val_losses:
            val_losses[key] /= num_batches

        # Compute metrics
        metrics = {'loss': val_losses['total']}

        # Classification metrics
        if all_predictions:
            all_predictions = torch.cat(all_predictions)
            all_labels = torch.cat(all_labels)
            cls_metrics = compute_classification_metrics(
                all_predictions.numpy(),
                all_labels.numpy()
            )
            metrics.update({f'classification/{k}': v for k, v in cls_metrics.items()})

        # Localization metrics
        if all_masks_pred:
            dice_scores = []
            iou_scores = []
            for pred, gt in zip(all_masks_pred, all_masks_gt):
                dice = compute_dice_score(pred, gt)
                iou = compute_iou(pred, gt)
                dice_scores.append(dice)
                iou_scores.append(iou)

            metrics['localization/dice'] = sum(dice_scores) / len(dice_scores)
            metrics['localization/iou'] = sum(iou_scores) / len(iou_scores)

        # Report generation metrics
        if all_reports_pred and all_reports_gt:
            report_metrics = compute_report_metrics(
                all_reports_pred,
                all_reports_gt
            )
            metrics.update({f'generation/{k}': v for k, v in report_metrics.items()})

        return metrics

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'phase': self.phase,
        }

        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"Saved best model: {best_path}")

    def train(self):
        """Main training loop."""
        print(f"\nStarting training for {self.max_epochs} epochs...")
        start_time = time.time()

        for epoch in range(1, self.max_epochs + 1):
            self.current_epoch = epoch

            # Train
            train_losses = self.train_epoch()

            # Log training losses
            print(f"\nEpoch {epoch}/{self.max_epochs}")
            print(f"Train Loss: {train_losses['total']:.4f}")

            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train/epoch_loss': train_losses['total'],
                    'train/classification_loss': train_losses['classification'],
                    'train/localization_loss': train_losses['localization'],
                    'train/generation_loss': train_losses['generation'],
                })

            # Validate
            if epoch % self.eval_interval == 0:
                val_metrics = self.validate()

                print(f"Val Loss: {val_metrics['loss']:.4f}")
                for key, value in val_metrics.items():
                    if key != 'loss':
                        print(f"  {key}: {value:.4f}")

                # Check if best model
                metric_key = 'classification/auc' if self.phase == 1 else 'localization/dice'
                current_metric = val_metrics.get(metric_key, 0.0)
                is_best = current_metric > self.best_metric

                if is_best:
                    self.best_metric = current_metric
                    print(f"New best {metric_key}: {self.best_metric:.4f}")

                # Save checkpoint
                if epoch % self.save_interval == 0 or is_best:
                    self.save_checkpoint(epoch, val_metrics, is_best)

                if self.use_wandb:
                    wandb.log({
                        'epoch': epoch,
                        'val/loss': val_metrics['loss'],
                        **{f'val/{k}': v for k, v in val_metrics.items() if k != 'loss'}
                    })

        elapsed = time.time() - start_time
        print(f"\nTraining completed in {elapsed/3600:.2f} hours")
        print(f"Best metric: {self.best_metric:.4f}")


def create_optimizer(
    model: nn.Module,
    learning_rate: float = 1e-4,
    vision_lr_multiplier: float = 0.1,
    weight_decay: float = 0.01
) -> AdamW:
    """
    Create optimizer with different learning rates for different components.

    Args:
        model: Model to optimize
        learning_rate: Base learning rate
        vision_lr_multiplier: Learning rate multiplier for vision encoder
        weight_decay: Weight decay

    Returns:
        AdamW optimizer
    """
    vision_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if 'vision_encoder' in name:
            vision_params.append(param)
        else:
            other_params.append(param)

    param_groups = [
        {'params': vision_params, 'lr': learning_rate * vision_lr_multiplier},
        {'params': other_params, 'lr': learning_rate},
    ]

    optimizer = AdamW(param_groups, weight_decay=weight_decay)

    print(f"Optimizer created:")
    print(f"  Vision encoder LR: {learning_rate * vision_lr_multiplier}")
    print(f"  Other components LR: {learning_rate}")

    return optimizer


if __name__ == "__main__":
    print("Trainer module loaded successfully")
