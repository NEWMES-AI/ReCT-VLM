"""
Main Training Script for ReCT-VLM Multi-task Model

Usage:
    python training/train_multitask.py --config configs/config_large.yaml --phase 1
    python training/train_multitask.py --config configs/config_large.yaml --phase 2
    python training/train_multitask.py --config configs/config_large.yaml --phase 3

Author: ReCT-VLM Team
"""

import argparse
import os
import sys
from pathlib import Path
import yaml
from typing import Dict, Any

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
import wandb

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.multi_task_model import VLM3DMultiTask
from training.dataset_multitask import CTRATEMultiTaskDataset, collate_fn
from training.trainer import MultiTaskTrainer, create_optimizer


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_distributed():
    """Setup distributed training."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])

        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)

        return True, rank, world_size, local_rank
    else:
        return False, 0, 1, 0


def create_datasets(config: Dict[str, Any], phase: int):
    """Create training and validation datasets."""
    data_config = config['data']

    train_dataset = CTRATEMultiTaskDataset(
        data_dir=data_config['data_dir'],
        labels_csv=data_config['train_labels_csv'],
        reports_json=data_config.get('train_reports_json', None) if phase == 3 else None,
        split='train',
        max_slices=data_config['max_slices'],
        target_size=tuple(data_config['target_size']),
        use_augmentation=True,
    )

    val_dataset = CTRATEMultiTaskDataset(
        data_dir=data_config['data_dir'],
        labels_csv=data_config['val_labels_csv'],
        reports_json=data_config.get('val_reports_json', None) if phase == 3 else None,
        split='valid',
        max_slices=data_config['max_slices'],
        target_size=tuple(data_config['target_size']),
        use_augmentation=False,
    )

    return train_dataset, val_dataset


def create_dataloaders(
    train_dataset,
    val_dataset,
    batch_size: int,
    num_workers: int,
    distributed: bool = False,
    rank: int = 0
):
    """Create data loaders."""
    train_sampler = None
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=dist.get_world_size(),
            rank=rank,
            shuffle=True
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    return train_loader, val_loader


def create_model(config: Dict[str, Any], phase: int, device: str) -> VLM3DMultiTask:
    """Create multi-task model."""
    model_config = config['model']

    model = VLM3DMultiTask(
        # Vision Encoder
        in_channels=model_config['in_channels'],
        patch_size=tuple(model_config['patch_size']),
        embed_dim=model_config['embed_dim'],
        depth=model_config['depth'],
        num_heads=model_config['num_heads'],
        mlp_ratio=model_config['mlp_ratio'],
        num_regions=model_config['num_regions'],
        # Classification
        num_classes=model_config['num_classes'],
        biobert_model=model_config['biobert_model'],
        # Localization
        localization_diseases=model_config['localization_diseases'],
        # Report Generation
        llm_model_name=model_config.get('llm_model_name', 'meta-llama/Llama-3.1-70B-Instruct'),
        lora_r=model_config.get('lora_r', 16),
        lora_alpha=model_config.get('lora_alpha', 32),
        lora_dropout=model_config.get('lora_dropout', 0.1),
        # Training phase
        enable_classification=(phase >= 1),
        enable_localization=(phase >= 2),
        enable_generation=(phase >= 3),
    )

    # Load pretrained weights if specified
    if phase > 1 and 'pretrained_checkpoint' in config:
        checkpoint_path = config['pretrained_checkpoint']
        if os.path.exists(checkpoint_path):
            print(f"Loading pretrained checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print("Pretrained weights loaded successfully")

    return model


def main():
    parser = argparse.ArgumentParser(description="Train ReCT-VLM Multi-task Model")
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config file'
    )
    parser.add_argument(
        '--phase',
        type=int,
        required=True,
        choices=[1, 2, 3],
        help='Training phase: 1 (Vision Encoder + Classification), '
             '2 (+ Localization), 3 (+ Report Generation)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (overrides config)'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Resume from checkpoint'
    )
    parser.add_argument(
        '--local_rank',
        type=int,
        default=-1,
        help='Local rank for distributed training'
    )
    parser.add_argument(
        '--no-wandb',
        action='store_true',
        help='Disable Weights & Biases logging'
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Setup distributed training
    distributed, rank, world_size, local_rank = setup_distributed()

    # Device
    if distributed:
        device = f"cuda:{local_rank}"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Only print on rank 0
    is_main_process = (rank == 0)

    if is_main_process:
        print("=" * 80)
        print(f"ReCT-VLM Multi-task Training - Phase {args.phase}")
        print("=" * 80)
        print(f"Config: {args.config}")
        print(f"Device: {device}")
        print(f"Distributed: {distributed}")
        if distributed:
            print(f"World size: {world_size}")
            print(f"Rank: {rank}")

    # Output directory
    output_dir = args.output_dir if args.output_dir else config['training']['output_dir']
    output_dir = Path(output_dir) / f"phase_{args.phase}"
    output_dir.mkdir(parents=True, exist_ok=True)

    if is_main_process:
        print(f"Output directory: {output_dir}")

    # Initialize wandb
    use_wandb = (not args.no_wandb) and is_main_process and config['training'].get('use_wandb', False)
    if use_wandb:
        wandb.init(
            project=config['training'].get('wandb_project', 'rect-vlm'),
            name=f"phase{args.phase}_{config['model']['embed_dim']}d",
            config={
                'phase': args.phase,
                **config
            }
        )

    # Create datasets
    if is_main_process:
        print("\nCreating datasets...")

    train_dataset, val_dataset = create_datasets(config, args.phase)

    if is_main_process:
        print(f"Train dataset: {len(train_dataset)} samples")
        print(f"Val dataset: {len(val_dataset)} samples")

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
        distributed=distributed,
        rank=rank
    )

    if is_main_process:
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")

    # Create model
    if is_main_process:
        print("\nCreating model...")

    model = create_model(config, args.phase, device)

    if is_main_process:
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params/1e6:.2f}M")
        print(f"Trainable parameters: {trainable_params/1e6:.2f}M")

    # Wrap model for distributed training
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True
        )

    # Create optimizer
    optimizer = create_optimizer(
        model,
        learning_rate=config['training']['learning_rate'],
        vision_lr_multiplier=config['training']['vision_lr_multiplier'],
        weight_decay=config['training']['weight_decay']
    )

    # Create scheduler
    scheduler = None
    if config['training'].get('use_scheduler', True):
        total_steps = len(train_loader) * config['training']['max_epochs']
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=config['training'].get('min_lr', 1e-6)
        )

    # Resume from checkpoint
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        if is_main_process:
            print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        if is_main_process:
            print(f"Resumed from epoch {start_epoch}")

    # Create trainer
    trainer = MultiTaskTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=output_dir,
        phase=args.phase,
        max_epochs=config['training']['max_epochs'],
        mixed_precision=config['training'].get('mixed_precision', True),
        gradient_clip=config['training'].get('gradient_clip', 1.0),
        log_interval=config['training'].get('log_interval', 10),
        eval_interval=config['training'].get('eval_interval', 1),
        save_interval=config['training'].get('save_interval', 5),
        use_wandb=use_wandb,
    )

    # Set starting epoch
    trainer.current_epoch = start_epoch

    # Train
    if is_main_process:
        print("\n" + "=" * 80)
        print("Starting training...")
        print("=" * 80 + "\n")

    try:
        trainer.train()
    except KeyboardInterrupt:
        if is_main_process:
            print("\nTraining interrupted by user")
    except Exception as e:
        if is_main_process:
            print(f"\nError during training: {e}")
        raise

    if is_main_process:
        print("\n" + "=" * 80)
        print("Training completed!")
        print("=" * 80)

    # Cleanup
    if use_wandb:
        wandb.finish()

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
