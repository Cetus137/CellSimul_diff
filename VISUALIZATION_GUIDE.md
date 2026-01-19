# Training Visualization Guide

The training pipeline now includes automatic visualization of training progress and model outputs.

## Features

### 1. Loss Curves
After each epoch, a graph showing training and validation losses is saved to:
```
visualizations/loss_curves.png
```

This plot includes:
- Training loss (blue line)
- Validation loss (red line)
- Automatic log scale for large loss variations
- Grid for easy reading

### 2. Model Output Visualization
After each epoch, example model outputs are saved to:
```
visualizations/epoch_XXXX.png
```

Each visualization shows 4 samples (or fewer if batch size is smaller) with 7 columns:
1. **Original**: The ground truth image
2. **Noisy (t=X)**: The image with noise added at timestep t
3. **Heatmap**: Cell centre heatmap (conditioning channel 1)
4. **Distance**: Distance-to-nearest-centre map (conditioning channel 2)
5. **Boundary**: Boundary likelihood map (conditioning channel 3)
6. **Pred. Noise**: The noise predicted by the model
7. **Denoised**: The denoised image (reconstructed from noisy image using predicted noise)

## Configuration

Visualization can be controlled in `configs/train.yaml`:

```yaml
training:
  # Visualization
  visualize: true  # Enable/disable visualization
  viz_dir: "visualizations"  # Output directory
```

## Viewing Results

During training, check the `visualizations/` directory:

```bash
# View latest loss curves
open visualizations/loss_curves.png

# View specific epoch output
open visualizations/epoch_0000.png
open visualizations/epoch_0010.png
```

## Performance Notes

- Visualization adds minimal overhead (~5-10 seconds per epoch)
- Images are saved at 100 DPI by default
- Uses non-interactive matplotlib backend (Agg) - no display required
- All visualizations are automatically saved to disk

## Interpreting the Outputs

### Good Signs:
- **Loss curves**: Smooth decrease in both train and val loss
- **Pred. Noise**: Should resemble random noise initially, become more structured over time
- **Denoised**: Should progressively improve to match the Original image

### Warning Signs:
- **Loss curves**: Divergence between train and val (overfitting)
- **Pred. Noise**: All zeros or very large values (model collapse)
- **Denoised**: Artifacts, blurriness, or missing structure

## Customization

To change the number of samples shown or DPI:

Edit `training/visualizer.py`:
```python
# In visualize_model_output():
num_samples: int = 4,  # Change this
dpi: int = 100  # Change this in __init__
```
