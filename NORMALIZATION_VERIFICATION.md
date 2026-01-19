# Normalization Pipeline Verification

This document verifies that the normalization pipeline is correctly implemented.

## ✅ Verification Checklist

### 1. Dataset Returns Clean x0 Only
**Status: VERIFIED**

Location: [datasets/centre_condition_dataset.py](datasets/centre_condition_dataset.py)

```python
def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    # Load clean image
    image = np.load(image_file).astype(np.float32)
    
    # Normalize to [-1, 1]
    if self.normalize_images:
        image = self._normalize_image(image)
    
    # Generate conditioning
    conditioning = generate_conditioning_maps(...)
    
    # Return clean image + conditioning (NO NOISE)
    return image, conditioning
```

**✓ No calls to:**
- `torch.randn_like()`
- `q_sample()`
- `add_noise()`

**✓ Noise is ONLY added in DDPM.compute_loss(), not in dataset**

---

### 2. Batch Statistics Printed to Console
**Status: IMPLEMENTED**

Location: [training/trainer.py](training/trainer.py#L197-L221)

**On first batch of each epoch, prints:**
```
============================================================
ACTUAL BATCH STATS (Epoch 0, Batch 0)
============================================================
x0 dtype: torch.float32, shape: (B, 1, H, W)
x0 min/max/mean/std: -0.xxxxx, 0.xxxxx, -0.xxxxx, 0.xxxxx

cond dtype: torch.float32, shape: (B, 3, H, W)
cond[0] min/max/mean/std: 0.000000, 1.000000, 0.xxxxx, 0.xxxxx
cond[1] min/max/mean/std: 0.000000, 1.000000, 0.xxxxx, 0.xxxxx
cond[2] min/max/mean/std: 0.000000, 1.000000, 0.xxxxx, 0.xxxxx

------------------------------------------------------------
EXPECTED VALUES:
  For training in [-1,1]: x0 min≈-1, max≈1, std~0.1-0.6
  For training in [0,1]:  x0 min≈0, max≈1
  BAD SIGNS: min≈max (tiny std) or values outside expected range
------------------------------------------------------------
```

**What to look for:**
- ✓ x0 min ≈ -1.0, max ≈ 1.0 (or close)
- ✓ x0 std in range [0.1, 0.6] (not tiny like 0.01)
- ✓ conditioning channels all in [0, 1]
- ✗ BAD: min ≈ 0.49, max ≈ 0.51 (indicates normalization collapsed)
- ✗ BAD: min/max way outside [-1, 1]

---

### 3. Visualization Uses Fixed vmin/vmax
**Status: VERIFIED**

Location: [training/visualizer.py](training/visualizer.py#L186-L231)

**All imshow calls use explicit limits:**
```python
# Convert from [-1,1] to [0,1] for display
images_display = to_zero_one(images_np, data_min, data_max)

# Plot with FIXED limits (prevents matplotlib auto-scaling)
axes[i, 0].imshow(images_display[i, 0], cmap='gray', vmin=0, vmax=1)
axes[i, 1].imshow(noisy_display[i, 0], cmap='gray', vmin=0, vmax=1)
axes[i, 2].imshow(conditioning_np[i, 0], cmap='hot', vmin=0, vmax=1)
# ... etc for all plots
```

**✓ Every imshow() has vmin=0, vmax=1**
**✓ No auto-scaling that would make noise look like signal**

---

## Complete Normalization Flow

### Data Loading → Training
```
Raw uint16 image
    ↓
[utils/normalization.py] percentile normalization
    ├─ p1 = percentile(img, 1%)
    ├─ p99 = percentile(img, 99%)
    ├─ clip to [p1, p99]
    └─ scale to [0, 1]
    ↓
[utils/normalization.py] to_minus_one_one()
    └─ x = 2*x - 1  →  [-1, 1]
    ↓
[DDPM.compute_loss()] add noise
    └─ x_t = √α̅_t · x_0 + √(1-α̅_t) · ε
    ↓
[DDPM.model] predict noise
    └─ ε_pred = UNet(x_t, t, cond)
    ↓
Loss = ||ε_pred - ε_true||²
```

### Training → Visualization
```
Model output in [-1, 1]
    ↓
[visualizer.py] to_zero_one()
    └─ (x + 1) / 2  →  [0, 1]
    ↓
[matplotlib] imshow(vmin=0, vmax=1)
    └─ Display with fixed scaling
```

---

## Quick Test

Run one epoch and check console output:

```bash
python scripts/train.py
```

**Look for:**
1. Batch stats printed at start of epoch
2. x0 min ≈ -1, max ≈ 1, reasonable std
3. No range violation warnings
4. Generated visualizations show clear structure (not noise/blank)

**If visualizations look weird but batch stats are correct:**
→ Problem is in visualization conversion, not data pipeline

**If batch stats show tiny std or wrong range:**
→ Problem is in dataset normalization

---

## Common Bugs Caught by Batch Stats

| Symptom | Cause |
|---------|-------|
| min≈0.49, max≈0.51 | Normalized to [0,1] but forgot to convert to [-1,1] |
| min≈0, max≈1 | Training in [0,1] but DDPM expects [-1,1] |
| min≈-0.01, max≈0.01 | Double-normalized or divided by wrong constant |
| std < 0.01 | All images the same / normalization collapsed |
| min < -2 or max > 2 | Normalization not applied at all |

