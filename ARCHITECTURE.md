# Data Flow Architecture

This document illustrates how data flows through the centre-conditioned diffusion pipeline.

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         RAW DATA INPUT                              │
│  Microscopy Images (*.tif) + Instance Masks (*.tif)                 │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ↓
┌─────────────────────────────────────────────────────────────────────┐
│                      PREPROCESSING STAGE                            │
│  scripts/preprocess_data.py                                         │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ 1. Extract centres from masks                               │   │
│  │    → preprocessing/extract_centres.py                       │   │
│  │    Input: Instance mask (H×W, int)                          │   │
│  │    Output: Centres (N×2, float) in (y,x) format            │   │
│  └─────────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ 2. Extract patches                                          │   │
│  │    → preprocessing/extract_patches.py                       │   │
│  │    Input: Image (H×W) + Centres (N×2)                       │   │
│  │    Output: Patches (256×256) + Patch centres (M×2)         │   │
│  │    Filter: Keep only patches with ≥3 cells                  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ 3. Create train/val/test splits                             │   │
│  │    → preprocessing/build_dataset.py                         │   │
│  │    Split: 80% train / 10% val / 10% test                    │   │
│  └─────────────────────────────────────────────────────────────┘   │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    PROCESSED DATA STORAGE                           │
│  data/processed/patches/{train,val,test}/                           │
│  Each patch: *_image.npy, *_centres.npy, *_meta.npz                 │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ↓
┌─────────────────────────────────────────────────────────────────────┐
│                      TRAINING STAGE                                 │
│  scripts/train.py                                                   │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ Data Loading (per batch)                                    │   │
│  │ → datasets/centre_condition_dataset.py                      │   │
│  │                                                             │   │
│  │ 1. Load image patch (1×256×256)                             │   │
│  │ 2. Load centres (M×2)                                       │   │
│  │ 3. Generate conditioning on-the-fly:                        │   │
│  │    → preprocessing/generate_condition_maps.py               │   │
│  │    Output: (3×256×256) conditioning tensor                  │   │
│  │    [0] = Centre heatmap                                     │   │
│  │    [1] = Distance map                                       │   │
│  │    [2] = Boundary map                                       │   │
│  │ 4. Apply augmentations (flip, rotate)                       │   │
│  │ 5. Randomly drop conditioning (p=0.1) for CFG              │   │
│  └─────────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ Forward Pass                                                │   │
│  │ → models/diffusion.py                                       │   │
│  │                                                             │   │
│  │ 1. Sample timestep t ~ Uniform(0, T)                        │   │
│  │ 2. Add noise: x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε             │   │
│  │ 3. Predict noise: ε̂ = UNet(x_t, t, conditioning)           │   │
│  │    → models/unet.py                                         │   │
│  │ 4. Compute loss: L = ||ε - ε̂||²                            │   │
│  └─────────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ Training Loop                                               │   │
│  │ → training/trainer.py                                       │   │
│  │                                                             │   │
│  │ - Backward pass + gradient clipping                         │   │
│  │ - Update EMA every 10 steps                                 │   │
│  │ - Mixed precision (AMP)                                     │   │
│  │ - Save checkpoint every 5000 steps                          │   │
│  │ - Validate every 2000 steps                                 │   │
│  └─────────────────────────────────────────────────────────────┘   │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    TRAINED MODEL CHECKPOINT                         │
│  checkpoints/best.pt                                                │
│  Contains: model weights, EMA, optimizer state, step counter        │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ↓
┌─────────────────────────────────────────────────────────────────────┐
│                      SAMPLING STAGE                                 │
│  scripts/sample.py                                                  │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ Input: Cell centres (N×2) or random generation              │   │
│  │        → User provides or randomly sample                   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ 1. Generate conditioning from centres                       │   │
│  │    → preprocessing/generate_condition_maps.py               │   │
│  │    Output: (3×256×256) conditioning tensor                  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ 2. Reverse diffusion (denoising)                            │   │
│  │    → sampling/sample_from_centres.py                        │   │
│  │    → models/diffusion.py                                    │   │
│  │                                                             │   │
│  │    x_T ~ N(0, I)  (random noise)                            │   │
│  │    for t = T to 1:                                          │   │
│  │      if CFG:                                                │   │
│  │        ε̂_cond = UNet(x_t, t, conditioning)                  │   │
│  │        ε̂_uncond = UNet(x_t, t, zeros)                       │   │
│  │        ε̂ = ε̂_uncond + w·(ε̂_cond - ε̂_uncond)               │   │
│  │      else:                                                  │   │
│  │        ε̂ = UNet(x_t, t, conditioning)                       │   │
│  │      x_{t-1} ~ p(x_{t-1} | x_t, ε̂)                         │   │
│  │    return x_0 (generated image)                             │   │
│  └─────────────────────────────────────────────────────────────┘   │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    GENERATED SAMPLES                                │
│  samples/*.png - Individual images with overlaid centres            │
│  samples/grid.png - Grid of multiple samples                        │
│  samples/metadata.json - Generation parameters                      │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    EVALUATION (Optional)                            │
│  Manual analysis or automated metrics                               │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ Power Spectrum Analysis                                     │   │
│  │ → evaluation/power_spectrum.py                              │   │
│  │ Compare spatial frequency content                           │   │
│  └─────────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ Boundary Profile Analysis                                   │   │
│  │ → evaluation/boundary_profiles.py                           │   │
│  │ Analyze intensity transitions across cell boundaries        │   │
│  └─────────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ Segmentation Consistency                                    │   │
│  │ → evaluation/segmentation_consistency.py                    │   │
│  │ Re-segment generated images and compare statistics          │   │
│  │ (cell count, area, circularity, NN distances)               │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

## Key Data Transformations

### Preprocessing: Mask → Centres
```
Instance Mask         Centres
┌──────────┐         ┌─────┐
│ 1 1 1 2 2│         │y  x │
│ 1 1 2 2 2│   →     ├─────┤
│ 3 3 3 2 2│         │25 15│  (cell 1)
│ 3 3 3 0 0│         │18 35│  (cell 2)
│ 0 0 0 0 0│         │35 12│  (cell 3)
└──────────┘         └─────┘
```

### Conditioning: Centres → Maps
```
Centres (N×2)        3-Channel Conditioning (3×H×W)
┌─────┐             
│y  x │             ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
├─────┤             │ · · ● · · · │ │ 5 4 3 2 1 0 │ │ · · · · · · │
│25 15│   →         │ · · · · · · │ │ 4 3 2 1 0 1 │ │ · ▓▓▓▓▓ · · │
│18 35│             │ · · · · ● · │ │ 3 2 1 0 1 2 │ │ · ▓▓▓▓▓ · · │
│35 12│             │ · · · · · · │ │ 2 1 0 1 2 3 │ │ · · · · · · │
└─────┘             │ · ● · · · · │ │ 1 0 1 2 3 4 │ │ · · · · · · │
                    └─────────────┘ └─────────────┘ └─────────────┘
                    Centre Heatmap   Distance Map    Boundary Map
```

### Training: Clean Image → Noisy → Predicted Noise
```
x_0 (clean)      x_t (noisy)       ε̂ (predicted)    Loss
┌────────┐      ┌────────┐        ┌────────┐       
│████░░░░│  →   │▓▓▓▓▒▒▒▒│  →     │░░░▒▒░░░│  →   L = ||ε - ε̂||²
│████░░░░│  +ε  │▓▓▓▓▒▒▒▒│  UNet  │░░░▒▒░░░│
│░░░░████│      │▒▒▒▒▓▓▓▓│        │▒▒░░░▒▒▒│
└────────┘      └────────┘        └────────┘
```

### Sampling: Noise → Clean Image
```
x_T (noise)     x_{T-1}         ...         x_0 (clean)
┌────────┐     ┌────────┐                   ┌────────┐
│▒▒▓▓▒░▓▒│  →  │▒▓▓▒░▓▓░│  →  ... →        │████░░░░│
│▓░▒▓▒░▓▒│     │▓▒▓▓░░▓▒│                   │████░░░░│
│▒▓░▒▓░▒▓│     │▒▓░▓▒▒▓░│                   │░░░░████│
└────────┘     └────────┘                   └────────┘
   1000 steps of denoising guided by conditioning
```

## Conditioning Workflow Detail

```
User Input: centres = [[25, 15], [18, 35], [35, 12]]
                              ↓
┌──────────────────────────────────────────────────────┐
│ generate_conditioning_maps()                         │
│                                                      │
│ 1. Centre Heatmap:                                   │
│    For each pixel (y,x):                             │
│      heatmap[y,x] = Σᵢ exp(-||[y,x] - cᵢ||²/2σ²)    │
│                                                      │
│ 2. Distance Map:                                     │
│    For each pixel (y,x):                             │
│      dist[y,x] = min_i ||[y,x] - cᵢ||               │
│                                                      │
│ 3. Boundary Map:                                     │
│    For each pixel (y,x):                             │
│      probs = softmax(-distances / temperature)      │
│      boundary[y,x] = entropy(probs)                  │
│                                                      │
│ Stack: conditioning = stack([heatmap, dist, boundary])│
└──────────────────────────────────────────────────────┘
                              ↓
         conditioning tensor (3×256×256)
         Ready for U-Net input
```

## File Count Summary

**Configuration**: 3 YAML files  
**Documentation**: 3 Markdown files + 1 .gitignore  
**Core Implementation**: 16 Python modules  
**Main Scripts**: 3 entry points  
**Total**: 26 files

---

This architecture ensures:
- ✅ **Modularity**: Each stage is independent
- ✅ **Reproducibility**: All random operations are seeded
- ✅ **Flexibility**: Easy to swap components
- ✅ **Clarity**: Data transformations are explicit
- ✅ **Efficiency**: On-the-fly conditioning generation
