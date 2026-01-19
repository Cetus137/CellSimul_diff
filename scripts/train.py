"""
Main training script for centre-conditioned diffusion models.
"""

import argparse
import yaml
import torch
import torch.optim as optim
from pathlib import Path
import logging
import numpy as np

# Add parent directory to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.unet import ConditionalUNet
from models.diffusion import DDPM
from datasets.centre_condition_dataset import get_dataloader
from training.trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_paths):
    """Load and merge configuration files."""
    config = {}
    for path in config_paths:
        with open(path, 'r') as f:
            config.update(yaml.safe_load(f))
    return config


def create_model(config):
    """Create U-Net and DDPM from configuration."""
    unet = ConditionalUNet(
        in_channels=config['unet']['in_channels'],
        out_channels=config['unet']['out_channels'],
        condition_channels=config['unet']['condition_channels'],
        base_channels=config['unet']['base_channels'],
        channel_multipliers=config['unet']['channel_multipliers'],
        num_res_blocks=config['unet']['num_res_blocks'],
        attention_resolutions=config['unet']['attention_resolutions'],
        num_heads=config['unet']['num_heads'],
        time_emb_dim=config['unet']['time_emb_dim'],
        num_groups=config['unet']['norm_groups'],
        dropout=config['unet']['dropout']
    )
    
    ddpm = DDPM(
        model=unet,
        timesteps=config['diffusion']['timesteps'],
        beta_schedule=config['diffusion']['beta_schedule'],
        beta_start=config['diffusion'].get('beta_start', 0.0001),
        beta_end=config['diffusion'].get('beta_end', 0.02),
        prediction_type=config['diffusion']['prediction_type'],
        loss_type=config['diffusion']['loss_type']
    )
    
    return ddpm


def create_optimizer(model, config):
    """Create optimizer from configuration."""
    opt_config = config['optimizer']
    
    if opt_config['type'] == 'AdamW':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=opt_config['learning_rate'],
            weight_decay=opt_config['weight_decay'],
            betas=opt_config['betas'],
            eps=opt_config['eps']
        )
    elif opt_config['type'] == 'Adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=opt_config['learning_rate'],
            weight_decay=opt_config.get('weight_decay', 0),
            betas=opt_config.get('betas', (0.9, 0.999)),
            eps=opt_config.get('eps', 1e-8)
        )
    else:
        raise ValueError(f"Unknown optimizer type: {opt_config['type']}")
    
    return optimizer


def create_scheduler(optimizer, config):
    """Create learning rate scheduler from configuration."""
    sched_config = config['scheduler']
    
    if sched_config['type'] == 'cosine':
        # Cosine annealing with warmup
        from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
        
        warmup_steps = sched_config['warmup_steps']
        total_steps = config['training']['num_epochs'] * 1000  # Approximate
        
        def warmup_cosine(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                return 0.5 * (1 + np.cos(np.pi * progress))
        
        scheduler = LambdaLR(optimizer, warmup_cosine)
    
    elif sched_config['type'] == 'constant':
        scheduler = None
    
    else:
        raise ValueError(f"Unknown scheduler type: {sched_config['type']}")
    
    return scheduler


def main():
    parser = argparse.ArgumentParser(
        description="Train centre-conditioned diffusion model"
    )
    parser.add_argument(
        '--data_config',
        type=str,
        default='configs/data.yaml',
        help='Path to data configuration'
    )
    parser.add_argument(
        '--model_config',
        type=str,
        default='configs/model.yaml',
        help='Path to model configuration'
    )
    parser.add_argument(
        '--train_config',
        type=str,
        default='configs/train.yaml',
        help='Path to training configuration'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to train on (overrides config)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config([
        args.data_config,
        args.model_config,
        args.train_config
    ])
    
    # Set device
    if args.device:
        device = args.device
    else:
        device = config['device']
    
    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = 'cpu'
    
    # Set random seed
    seed = config['random_seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    logger.info("="*60)
    logger.info("Starting training")
    logger.info("="*60)
    logger.info(f"Device: {device}")
    logger.info(f"Random seed: {seed}")
    
    # Create data loaders
    logger.info("Creating data loaders...")
    
    train_loader = get_dataloader(
        patches_dir=config['processed_data']['patches_dir'],
        split='train',
        batch_size=config['dataloader']['batch_size'],
        num_workers=config['dataloader']['num_workers'],
        heatmap_sigma=config['preprocessing']['centre_heatmap_sigma'],
        augment=config['augmentation']['random_flip_horizontal'],  # Enable augmentation
        p_uncond=config['diffusion']['cfg'].get('dropout_prob', 0.0),
        shuffle=True
    )
    
    val_loader = get_dataloader(
        patches_dir=config['processed_data']['patches_dir'],
        split='val',
        batch_size=config['dataloader']['batch_size'],
        num_workers=config['dataloader']['num_workers'],
        heatmap_sigma=config['preprocessing']['centre_heatmap_sigma'],
        augment=False,
        p_uncond=0.0,
        shuffle=False
    )
    
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")
    
    # Create model
    logger.info("Creating model...")
    model = create_model(config)
    
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    
    logger.info(f"Optimizer: {config['optimizer']['type']}")
    logger.info(f"Learning rate: {config['optimizer']['learning_rate']:.2e}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=config['training']['checkpoint_dir'],
        use_ema=config['training']['ema']['enabled'],
        ema_decay=config['training']['ema']['decay'],
        ema_update_every=config['training']['ema']['update_every'],
        mixed_precision=config['training']['mixed_precision'],
        gradient_clip=config['training']['gradient_clip'],
        log_every=config['training']['log_every'],
        save_every=config['training']['save_every'],
        validate_every=config['training']['validate_every'],
        visualize=config['training'].get('visualize', True),
        viz_dir=config['training'].get('viz_dir', 'visualizations')
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train
    logger.info("="*60)
    logger.info("Starting training loop")
    logger.info("="*60)
    
    try:
        trainer.train(num_epochs=config['training']['num_epochs'])
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        trainer.save_checkpoint(name='interrupted.pt')
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    
    logger.info("="*60)
    logger.info("Training completed!")
    logger.info("="*60)
    logger.info(f"Checkpoints saved to: {config['training']['checkpoint_dir']}")
    logger.info("Next steps:")
    logger.info("  1. Evaluate the model")
    logger.info("  2. Generate samples: python scripts/sample.py --checkpoint <path>")


if __name__ == "__main__":
    main()
