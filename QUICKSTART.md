# Quick Start Guide

This guide will get you up and running with the centre-conditioned diffusion model.

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended, 8GB+ VRAM)
- 20GB+ disk space for processed data

## Step-by-Step Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Your Data

Place your microscopy images and instance masks in:

```
data/raw/images/     # Your .tif or .png microscopy images
data/raw/masks/      # Corresponding instance masks
```

**Important**: Mask filenames must have `masks_` prefix matching the image filename:
- Image: `slice_001_XY_z90.tif`
- Mask: `masks_slice_001_XY_z90.tif`

### 3. Preprocess Data

```bash
python scripts/preprocess_data.py
```

This will:
- Extract cell centres from masks
- Create 256×256 patches with overlap
- Split into train/val/test (80/10/10)
- Save to `data/processed/patches/`

**Expected output**: ~1000-10000 patches depending on your dataset size

### 4. Train the Model

```bash
python scripts/train.py
```

**Training time**: ~12-24 hours on a single GPU for 100k steps

**Monitor progress**:
- Checkpoints saved to `checkpoints/`
- Loss logged every 100 steps
- Validation every 2000 steps

**Tip**: Start with a smaller model for testing:
Edit `configs/model.yaml` and set `base_channels: 64` (instead of 128)

### 5. Generate Samples

After training, generate samples:

```bash
python scripts/sample.py \
    --checkpoint checkpoints/best.pt \
    --num_samples 16 \
    --use_cfg \
    --guidance_scale 3.0 \
    --output_dir samples/
```

Samples will be saved in `samples/` with conditioning maps overlaid.

## Configuration Tips

### For Faster Experimentation

Edit `configs/train.yaml`:
```yaml
training:
  num_epochs: 50  # Reduce from 500
  save_every: 1000  # Save less frequently
```

Edit `configs/model.yaml`:
```yaml
unet:
  base_channels: 64  # Reduce from 128
diffusion:
  timesteps: 500  # Reduce from 1000 (faster sampling)
```

### For Better Quality (if you have time/compute)

Edit `configs/model.yaml`:
```yaml
unet:
  base_channels: 192  # Increase from 128
  num_res_blocks: 3  # Increase from 2
diffusion:
  beta_schedule: "cosine"  # Often better than linear
```

Edit `configs/train.yaml`:
```yaml
optimizer:
  learning_rate: 1.0e-4  # Lower LR for stability
training:
  num_epochs: 1000  # Train longer
```

## Common Issues

### 1. Out of Memory

**Solution**: Reduce batch size in `configs/data.yaml`:
```yaml
dataloader:
  batch_size: 4  # Reduce from 8
```

Or reduce model size (see above).

### 2. No Patches Generated

**Check**:
- Are images and masks in correct directories?
- Do filenames match (with `_mask` suffix)?
- Are there enough cells per image? (min 3 required)

**Solution**: Lower `min_cells_per_patch` in `configs/data.yaml`

### 3. Poor Sample Quality

**Try**:
1. Train longer (check if loss is still decreasing)
2. Use classifier-free guidance at higher scales (3.0-7.0)
3. Ensure conditioning maps look reasonable (check visualizations)
4. Verify your training data quality

### 4. Samples Don't Match Centres

**Possible causes**:
- Model undertrained
- Conditioning dropout too high (reduce in `configs/model.yaml`)
- Wrong conditioning parameters (check sigma values)

## What to Expect

### Training Progress

- **Steps 0-10k**: Noisy outputs, basic structure emerging
- **Steps 10k-50k**: Recognizable cells, improving texture
- **Steps 50k-100k**: Realistic morphology, fine details
- **Steps 100k+**: Diminishing returns, subtle refinements

### Sample Quality Indicators

**Good signs**:
- Cells roughly centered on provided coordinates
- Realistic membrane/boundary appearance
- Appropriate cell sizes and shapes
- No major artifacts (holes, disconnections)

**Warning signs**:
- Cells far from input centres → Increase conditioning strength
- Uniform/blob-like cells → Train longer or adjust model
- Grainy texture → Check data normalization

## Next Steps

Once you have a working model:

1. **Evaluate quantitatively**:
   ```python
   # See evaluation/ modules
   from evaluation.power_spectrum import compare_power_spectra
   from evaluation.segmentation_consistency import compare_statistics
   ```

2. **Experiment with conditioning**:
   - Try different sigma values
   - Modify boundary map generation
   - Add custom conditioning channels

3. **Scale up**:
   - Train on larger patches (512×512)
   - Use more training data
   - Increase model capacity

4. **Deploy**:
   - Export lightweight sampling code
   - Optimize inference speed (fewer timesteps, DDIM)
   - Create interactive demo

## Getting Help

- Check README.md for detailed documentation
- Review code comments for implementation details
- Open an issue for bugs or questions

## Minimal Working Example

If you just want to test the pipeline with dummy data:

```python
# Create synthetic data
import numpy as np
import tifffile

# Generate 10 random cell images
for i in range(10):
    # Random cell centers
    centres = np.random.rand(20, 2) * 200
    
    # Create simple Voronoi-like image
    from scipy.spatial import distance_matrix
    y, x = np.ogrid[:256, :256]
    coords = np.stack([y.ravel(), x.ravel()], axis=1)
    dists = distance_matrix(coords, centres)
    mask = dists.argmin(axis=1).reshape(256, 256) + 1
    
    # Save
    tifffile.imwrite(f'data/raw/images/test_{i:03d}.tif', 
                (np.random.rand(256, 256) * 255).astype(np.uint8))
    tifffile.imwrite(f'data/raw/masks/masks_test_{i:03d}.tif', 
                mask.astype(np.uint16))

# Then run preprocessing and training as normal
```

---

Happy diffusing! 🔬✨
