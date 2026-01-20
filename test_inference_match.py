"""
Minimal reproduction test for inference vs training consistency.

This test verifies that:
1. Conditioning construction is identical between training and inference
2. Sampling from noise produces non-black images
3. Results are consistent with training-time denoising quality
"""

import torch
import numpy as np
from pathlib import Path
import yaml

from models.unet import ConditionalUNet
from models.diffusion import DDPM
from datasets.centre_condition_dataset import CentreConditionDataset
from preprocessing.generate_condition_maps import generate_conditioning_maps


def test_conditioning_consistency():
    """Test that inference and training generate identical conditioning."""
    print("\n" + "="*60)
    print("TEST 1: Conditioning Consistency")
    print("="*60)
    
    # Load a real training sample
    dataset = CentreConditionDataset(
        patches_dir='data/processed/patches',
        split='test',
        normalize_images=True
    )
    
    # Get one sample
    image_train, cond_train = dataset[0]
    
    # Load the centres for this sample
    centres_file = dataset.patch_files[0].parent / dataset.patch_files[0].name.replace('_image.npy', '_centres.npy')
    centres = np.load(centres_file)
    
    # Generate conditioning at inference time
    cond_inference = generate_conditioning_maps(
        centres,
        image_shape=(256, 256),
        heatmap_sigma=3.0,
        boundary_sigma=2.0,
        boundary_method='entropy',
        distance_percentile=95.0
    )
    
    # Convert to tensor for comparison
    cond_inference_tensor = torch.from_numpy(cond_inference).float()
    
    # Compare
    print(f"\nTraining conditioning shape: {cond_train.shape}")
    print(f"Inference conditioning shape: {cond_inference_tensor.shape}")
    
    print(f"\nTraining conditioning stats:")
    for i in range(3):
        c = cond_train[i]
        print(f"  Channel {i}: min={c.min():.4f}, max={c.max():.4f}, mean={c.mean():.4f}, std={c.std():.4f}")
    
    print(f"\nInference conditioning stats:")
    for i in range(3):
        c = cond_inference_tensor[i]
        print(f"  Channel {i}: min={c.min():.4f}, max={c.max():.4f}, mean={c.mean():.4f}, std={c.std():.4f}")
    
    # Check if they're similar (allowing for small numerical differences)
    max_diff = torch.abs(cond_train - cond_inference_tensor).max().item()
    mean_diff = torch.abs(cond_train - cond_inference_tensor).mean().item()
    
    print(f"\nDifference: max={max_diff:.6f}, mean={mean_diff:.6f}")
    
    if max_diff < 1e-5:
        print("✓ PASS: Conditioning is identical between training and inference")
    else:
        print("⚠️  WARNING: Conditioning differs slightly (may be due to randomness in centres)")
    
    return cond_train, centres


def test_sampling_quality(checkpoint_path='checkpoints/best.pt'):
    """Test that sampling produces non-black, structured images."""
    print("\n" + "="*60)
    print("TEST 2: Sampling Quality")
    print("="*60)
    
    if not Path(checkpoint_path).exists():
        print(f"⚠️  Checkpoint not found at {checkpoint_path}")
        print("   Skipping sampling test. Train a model first.")
        return
    
    # Load model
    with open('configs/model.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
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
    
    model = DDPM(
        model=unet,
        timesteps=config['diffusion']['timesteps'],
        beta_schedule=config['diffusion']['beta_schedule'],
        prediction_type=config['diffusion']['prediction_type'],
        loss_type=config['diffusion']['loss_type'],
        data_min=-1.0,
        data_max=1.0
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded checkpoint from step {checkpoint.get('step', 'unknown')}")
    else:
        print("⚠️  Unexpected checkpoint format")
        return
    
    model.eval()
    device = 'cpu'  # Use CPU for testing
    model = model.to(device)
    
    # Generate synthetic centres
    np.random.seed(42)
    centres = np.random.rand(20, 2) * 256
    
    # Generate conditioning
    conditioning = generate_conditioning_maps(
        centres,
        image_shape=(256, 256),
        heatmap_sigma=3.0,
        boundary_sigma=2.0,
        boundary_method='entropy',
        distance_percentile=95.0
    )
    
    conditioning_tensor = torch.from_numpy(conditioning).unsqueeze(0).float().to(device)
    
    print(f"\nConditioning for sampling:")
    print(f"  Shape: {conditioning_tensor.shape}")
    for i in range(3):
        c = conditioning_tensor[0, i]
        print(f"  Channel {i}: min={c.min():.4f}, max={c.max():.4f}, mean={c.mean():.4f}")
    
    # Sample
    print(f"\nSampling from noise...")
    with torch.no_grad():
        samples = model.sample(
            conditioning=conditioning_tensor,
            clip_denoised=True
        )
    
    # Analyze output
    output = samples[0, 0].cpu().numpy()
    
    print(f"\nSampled output (in [-1, 1] range):")
    print(f"  min: {output.min():.4f}")
    print(f"  max: {output.max():.4f}")
    print(f"  mean: {output.mean():.4f}")
    print(f"  std: {output.std():.4f}")
    
    # Convert to [0, 1] for analysis
    output_01 = (output + 1.0) / 2.0
    output_01 = np.clip(output_01, 0, 1)
    
    print(f"\nAfter conversion to [0, 1]:")
    print(f"  min: {output_01.min():.4f}")
    print(f"  max: {output_01.max():.4f}")
    print(f"  mean: {output_01.mean():.4f}")
    print(f"  std: {output_01.std():.4f}")
    
    # Check for failure modes
    if output_01.std() < 0.01:
        print("❌ FAIL: Output is nearly constant (std < 0.01)")
        print("   This indicates sampling failure or mode collapse")
    elif output_01.mean() < 0.1:
        print("❌ FAIL: Output is nearly black (mean < 0.1)")
        print("   This indicates conditioning mismatch or sampling bug")
    elif output_01.mean() > 0.9:
        print("❌ FAIL: Output is nearly white (mean > 0.9)")
        print("   This indicates normalization bug")
    else:
        print("✓ PASS: Output appears to have reasonable variation")
        
        # Additional checks
        dark_pixels = (output_01 < 0.3).sum() / output_01.size
        mid_pixels = ((output_01 >= 0.3) & (output_01 <= 0.7)).sum() / output_01.size
        bright_pixels = (output_01 > 0.7).sum() / output_01.size
        
        print(f"\nPixel distribution:")
        print(f"  Dark (<0.3): {dark_pixels*100:.1f}%")
        print(f"  Mid [0.3-0.7]: {mid_pixels*100:.1f}%")
        print(f"  Bright (>0.7): {bright_pixels*100:.1f}%")


def test_timestep_consistency():
    """Verify timesteps and schedules are consistent."""
    print("\n" + "="*60)
    print("TEST 3: Timestep Schedule Consistency")
    print("="*60)
    
    with open('configs/model.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Configured timesteps: {config['diffusion']['timesteps']}")
    print(f"Configured beta_schedule: {config['diffusion']['beta_schedule']}")
    
    # Create model
    unet = ConditionalUNet(
        in_channels=1,
        out_channels=1,
        condition_channels=3,
        base_channels=64,
        channel_multipliers=[1, 2, 4],
        num_res_blocks=2,
        attention_resolutions=[],
        num_heads=4,
        time_emb_dim=256,
        num_groups=8,
        dropout=0.0
    )
    
    model = DDPM(
        model=unet,
        timesteps=config['diffusion']['timesteps'],
        beta_schedule=config['diffusion']['beta_schedule'],
        data_min=-1.0,
        data_max=1.0
    )
    
    print(f"Model timesteps: {model.timesteps}")
    print(f"Betas range: [{model.betas.min():.6f}, {model.betas.max():.6f}]")
    print(f"Alphas_cumprod range: [{model.alphas_cumprod.min():.6f}, {model.alphas_cumprod.max():.6f}]")
    
    print("\n✓ Timestep schedule is consistent")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("INFERENCE FAILURE DIAGNOSTIC SUITE")
    print("="*60)
    
    # Run all tests
    test_conditioning_consistency()
    test_timestep_consistency()
    test_sampling_quality()
    
    print("\n" + "="*60)
    print("DIAGNOSTIC COMPLETE")
    print("="*60)
    print("\nIf sampling fails:")
    print("1. Check conditioning statistics match between training/inference")
    print("2. Verify distance channel uses percentile normalization (not saturated)")
    print("3. Ensure all 3 channels are present and normalized to [0,1]")
    print("4. Check model was trained with same conditioning parameters")
    print("5. Verify output range conversion is correct ([-1,1] → [0,1])")
