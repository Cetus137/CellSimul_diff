# Centre-Conditioned Diffusion for Microscopy Image Synthesis

A research-grade PyTorch implementation of centre-conditioned diffusion models for generating realistic membrane-like microscopy images from cell centre locations only.

## Overview

This repository implements a conditional DDPM (Denoising Diffusion Probabilistic Model) that learns to generate synthetic microscopy images conditioned exclusively on cell centre positions. Unlike mask-to-image translation, this approach generates realistic cell morphologies and textures from minimal geometric priors.

**Key features:**
- **Centre-only conditioning**: No cell shape information required at generation time
- **Multi-channel geometric conditioning**: Combines centre heatmaps, distance fields, and boundary likelihood maps
- **Classifier-free guidance**: Improved sample quality through guidance scaling
- **Production-ready**: Modular architecture, comprehensive evaluation, mixed-precision training

## Problem Definition

**Given:**
- Raw microscopy images (single-channel fluorescence)
- Instance segmentation masks (each cell has unique ID)

**Goal:**
- Train a model that generates realistic microscopy images from cell centre coordinates only
- Learn realistic cell morphologies, textures, and spatial relationships

**Key constraint:**
- Masks are used ONLY during preprocessing to extract centres
- At inference, only centres are needed (not masks)

## Conceptual Model

### Conditioning Representation

The model is conditioned on three rasterized geometry maps derived from cell centres:

1. **Centre Heatmap** (`C₀`): Gaussian blobs at each centre
   - `C₀(x,y) = Σᵢ exp(-||x - cᵢ||² / 2σ²)`
   - Provides soft localization of cell positions

2. **Distance Map** (`C₁`): Euclidean distance to nearest centre
   - `C₁(x,y) = min_i ||x - cᵢ||`
   - Encodes spatial organization and cell density

3. **Boundary Likelihood Map** (`C₂`): Soft assignment entropy
   - Derived from distance-based probabilistic assignment
   - High values indicate equidistance from multiple centres (likely boundaries)

These are stacked into a 3-channel conditioning tensor: **`c ∈ ℝ^(H×W×3)`**

### Diffusion Model Architecture

- **Base model**: DDPM with ε-prediction (noise prediction)
- **Backbone**: Conditional U-Net with:
  - Time embedding via sinusoidal encoding
  - Conditioning injection via channel concatenation
  - Self-attention at mid-resolutions
  - GroupNorm, residual blocks, skip connections
- **Training**: Randomly drop conditioning (p=0.1) for classifier-free guidance
- **Sampling**: Optional CFG with guidance scale w

## Repository Structure

```
cell-centre-diffusion/
│
├── configs/                    # YAML configuration files
│   ├── data.yaml              # Data preprocessing and loading
│   ├── model.yaml             # U-Net and diffusion architecture
│   └── train.yaml             # Training hyperparameters
│
├── data/                       # Data storage
│   ├── raw/
│   │   ├── images/            # Place raw microscopy images here
│   │   └── masks/             # Place instance masks here
│   ├── processed/
│   │   ├── patches/           # Extracted patches (train/val/test)
│   │   └── metadata/          # Dataset index and metadata
│   └── splits/                # Train/val/test split indices
│
├── preprocessing/              # Data preprocessing modules
│   ├── extract_centres.py     # Centroid extraction from masks
│   ├── generate_condition_maps.py  # Generate 3-channel conditioning
│   ├── extract_patches.py     # Patch extraction with centres
│   └── build_dataset.py       # Full preprocessing pipeline
│
├── datasets/                   # PyTorch datasets
│   └── centre_condition_dataset.py  # Dataset with on-the-fly conditioning
│
├── models/                     # Neural network modules
│   ├── unet.py                # Conditional U-Net architecture
│   └── diffusion.py           # DDPM implementation
│
├── training/                   # Training infrastructure
│   ├── trainer.py             # Training loop with EMA, mixed precision
│   └── losses.py              # Loss functions
│
├── sampling/                   # Sampling and generation
│   └── sample_from_centres.py # Sample images from centres
│
├── evaluation/                 # Evaluation metrics
│   ├── power_spectrum.py      # Spatial frequency analysis
│   ├── boundary_profiles.py   # Intensity profile comparison
│   └── segmentation_consistency.py  # Re-segment and compare statistics
│
├── scripts/                    # Main entry points
│   ├── preprocess_data.py     # Run preprocessing pipeline
│   ├── train.py               # Train diffusion model
│   └── sample.py              # Generate samples from trained model
│
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- CUDA (recommended for training)

### Setup

```bash
# Clone repository
cd cell_simul

# Install dependencies
pip install -r requirements.txt
```

## Data Preparation

### 1. Prepare Raw Data

Place your data in the following structure:

```
data/raw/
├── images/
│   ├── image_001.tif
│   ├── image_002.tif
│   └── ...
└── masks/
    ├── image_001_mask.tif
    ├── image_002_mask.tif
    └── ...
```

**Requirements:**
- Images: Single-channel microscopy images (`.tif`, `.png`, etc.)
- Masks: Instance segmentation masks with unique integer ID per cell
- Filenames: Mask filename must have `masks_` prefix matching image filename
  - Example: `slice_001_XY_z90.tif` → `masks_slice_001_XY_z90.tif`

### 2. Run Preprocessing

```bash
python scripts/preprocess_data.py --config configs/data.yaml
```

This will:
1. Extract cell centres from each mask
2. Generate overlapping patches (default 256×256)
3. Filter patches by minimum cell count
4. Create train/val/test splits (80/10/10)
5. Save processed patches to `data/processed/patches/`

**Configuration** (`configs/data.yaml`):
- `patch_size`: Size of extracted patches (default: 256)
- `patch_stride`: Overlap between patches (default: 128)
- `min_cells_per_patch`: Minimum cells required (default: 3)
- `centre_heatmap_sigma`: Gaussian sigma for heatmap (default: 3.0)

## Training

### Basic Training

```bash
python scripts/train.py
```

This uses default configurations from `configs/*.yaml`.

### Advanced Options

```bash
python scripts/train.py \
    --data_config configs/data.yaml \
    --model_config configs/model.yaml \
    --train_config configs/train.yaml \
    --device cuda
```

### Resume Training

```bash
python scripts/train.py --resume checkpoints/checkpoint_step_10000.pt
```

### Key Training Settings

**Model** (`configs/model.yaml`):
- `base_channels`: U-Net base width (default: 128)
- `timesteps`: Number of diffusion steps (default: 1000)
- `beta_schedule`: Noise schedule (linear, cosine, quadratic)
- `cfg.dropout_prob`: Conditioning dropout for CFG (default: 0.1)

**Training** (`configs/train.yaml`):
- `learning_rate`: Initial LR (default: 2e-4)
- `num_epochs`: Training epochs (default: 500)
- `batch_size`: Batch size (default: 8)
- `ema.enabled`: Use exponential moving average (default: true)
- `mixed_precision`: Use AMP (default: true)

### Monitoring

Training logs include:
- Loss curves (train/validation)
- Learning rate schedule
- Gradient norms
- EMA updates

Checkpoints are saved to `checkpoints/` every N steps (configurable).

## Sampling

### Generate Samples

```bash
python scripts/sample.py \
    --checkpoint checkpoints/best.pt \
    --num_samples 16 \
    --num_cells 20 \
    --use_cfg \
    --guidance_scale 3.0 \
    --output_dir samples/
```

### Options

- `--checkpoint`: Path to trained model checkpoint
- `--centres`: Optional `.npy` file with specific centres (generates random if not provided)
- `--num_samples`: Number of images to generate
- `--num_cells`: Cells per image (if generating random centres)
- `--use_cfg`: Enable classifier-free guidance
- `--guidance_scale`: CFG strength (typical: 1.5-5.0)
- `--save_grid`: Save samples as a grid image

### Output

For each sample, the script saves:
- Individual image with overlaid centres
- Conditioning maps (heatmap, distance, boundary)
- Optional: Grid of all samples
- Metadata JSON

## Evaluation

### Power Spectrum Analysis

Compare spatial frequency content of real vs. generated images:

```python
from evaluation.power_spectrum import compare_power_spectra

distance = compare_power_spectra(
    real_images,
    generated_images,
    save_path='power_spectrum.png'
)
```

**Interpretation**: Lower L2 distance indicates better texture matching.

### Boundary Profiles

Analyze intensity transitions across cell boundaries:

```python
from evaluation.boundary_profiles import compare_boundary_profiles

distance = compare_boundary_profiles(
    real_images, real_centres_list,
    generated_images, gen_centres_list,
    save_path='boundary_profiles.png'
)
```

**Interpretation**: Assesses whether cell-cell interfaces are realistic.

### Segmentation Consistency

Re-segment generated images and compare cell statistics:

```python
from evaluation.segmentation_consistency import compare_statistics

distances = compare_statistics(
    real_stats_list,
    gen_stats_list,
    save_path='segmentation_stats.png'
)
```

**Metrics**:
- Cell count distribution
- Cell area distribution (Wasserstein distance)
- Circularity distribution
- Nearest-neighbor distances

## Architecture Details

### U-Net Backbone

- **Encoder**: 4 levels with residual blocks and downsampling
- **Bottleneck**: Residual blocks + self-attention
- **Decoder**: 4 levels with skip connections and upsampling
- **Attention**: Multi-head self-attention at mid-resolutions (16×16, 8×8)
- **Normalization**: GroupNorm (32 groups)
- **Time embedding**: Sinusoidal encoding → MLP → injected into each ResBlock

### Diffusion Process

**Forward (training)**:
```
x_t = √(ᾱ_t) · x_0 + √(1 - ᾱ_t) · ε
```

**Reverse (sampling)**:
```
x_{t-1} ~ N(μ_θ(x_t, t, c), σ_t²)
```

**Classifier-Free Guidance**:
```
ε̃ = ε_uncond + w · (ε_cond - ε_uncond)
```

where w is the guidance scale.

## Assumptions and Limitations

### Assumptions

1. **Cell centres are deterministic**: Given a mask, centroid computation is reproducible
2. **Spatial relationships matter**: Cell layout influences morphology
3. **Centre geometry is sufficient**: Voronoi-like tessellation provides implicit shape priors
4. **Single modality**: Model trained on one imaging modality

### Limitations

1. **Non-deterministic shapes**: Same centres can yield different cell shapes (by design)
2. **No explicit shape control**: Cannot specify "make this cell round" without retraining
3. **Boundary artifacts**: Rare cases of disconnected regions or overlaps
4. **Resolution dependency**: Trained patch size (256×256) is fixed
5. **Cell count variance**: Generated images may have slightly different cell counts than conditioning

### When This Approach Works Best

- High cell density (10+ cells per patch)
- Relatively uniform cell sizes
- Clear membrane/boundary signal in real data
- Stochastic shape variation is acceptable

### When to Consider Alternatives

- Need deterministic, controllable shapes → Use mask-to-image translation
- Very sparse cells (< 5 per image) → May need more shape information
- Multi-class segmentation → Extend conditioning to include cell types

## Technical Details

### Conditioning Map Generation

**Centre Heatmap**:
```python
heatmap = Σᵢ exp(-||coords - centre_i||² / (2σ²))
```

**Distance Map**:
```python
dist_map = min_i ||coords - centre_i||
```

**Boundary Map** (soft assignment entropy):
```python
probs = softmax(-distances / temperature)
boundary = -Σ probs · log(probs)
```

### Data Augmentation

Applied during training:
- Random horizontal flip
- Random vertical flip
- Random 90° rotation

Geometric consistency maintained between image and conditioning.

### Mixed Precision Training

Uses PyTorch AMP (`torch.cuda.amp`) for:
- Faster training (1.5-2× speedup)
- Reduced memory usage
- Maintained numerical stability via GradScaler

## Reproducibility

### Random Seeds

Set in all scripts via:
```python
torch.manual_seed(seed)
np.random.seed(seed)
```

**Note**: Diffusion sampling is inherently stochastic. For deterministic generation:
1. Fix random seed
2. Use same conditioning
3. Use DDIM sampling (TODO: not yet implemented)

### Checkpointing

Checkpoints include:
- Model weights
- Optimizer state
- EMA shadow parameters
- Training step counter
- Random number generator states (for exact resumption)

## Citation

If you use this code in your research, please cite:

```bibtex
@software{centre_diffusion_2026,
  title={Centre-Conditioned Diffusion Models for Microscopy Image Synthesis},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/cell-centre-diffusion}
}
```

## Acknowledgments

This implementation builds on:
- DDPM paper: [Ho et al. 2020](https://arxiv.org/abs/2006.11239)
- Classifier-free guidance: [Ho & Salimans 2021](https://arxiv.org/abs/2207.12598)
- U-Net architecture: [Ronneberger et al. 2015](https://arxiv.org/abs/1505.04597)

## License

MIT License - see LICENSE file for details.

## Contact

For questions or issues, please open a GitHub issue or contact [your email].

---

**Note**: This is research code. While production-quality in structure, expect rough edges and areas for optimization. Contributions welcome!
