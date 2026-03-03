#!/bin/bash
#SBATCH --job-name=timeseries_gen
#SBATCH --output=slogs/timeseries_%j.out
#SBATCH --error=slogs/timeseries_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu_short
#SBATCH --gpus-per-node=1
#SBATCH --constraint="a100|rtx8000"
#SBATCH --mem=32G

# Cell Diffusion Model - Time-Series Generation
# Generates synthetic time-series where cell centres undergo small displacements
# Simulates cell migration or temporal dynamics in microscopy

# Create logs directory if it doesn't exist
mkdir -p slogs

# Load required modules (adjust for your HPC)
module purge
source /well/kir/config/modules.sh
module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/12.0
source ~/devel/venv/Python-3.10.8-GCCcore-12.2.0/cell_simul_env/bin/activate

# Configuration
CHECKPOINT="/users/kir-fritzsche/aif490/devel/tissue_analysis/CellSimul_diff/checkpoints/long_training/best.pt"
OUTPUT_DIR="/users/kir-fritzsche/aif490/devel/tissue_analysis/CellSimul_diff/timeseries_output"
PREFIX="timeseries"

# Initial centres configuration
METHOD="from_file"  # Options: simple, poisson, training_dist, from_file
CENTRES_FILE="/users/kir-fritzsche/aif490/devel/tissue_analysis/CellSimul_diff/data/processed/test/patch_00000_centres.npy"

# Alternative: generate initial centres
# METHOD="poisson"
# DENSITY=0.0003
# MIN_DISTANCE=24.0

# Time-series parameters
NUM_TIMEPOINTS=10            # Number of timepoints to generate
DISPLACEMENT_STD=2        # Standard deviation of displacement per timestep (pixels) - SET TO 0 FOR STATIC CELLS
DISPLACEMENT_MEAN=0.0       # Mean displacement magnitude
MIN_DISTANCE=20.0           # Minimum distance between centres (pixels)
NOISE_CORRELATION=0.95      # Spatial correlation between consecutive frames (0-1, higher = more similar)
TEMPORAL_SMOOTHNESS=0.95     # AR(1) smoothing for noise evolution (0-1, higher = smoother transitions)
MATCH_HISTOGRAMS=True        # Apply histogram matching to ensure consistent intensity across frames (True/False)
# DIRECTIONAL_BIAS="0.5 0.3"  # Optional: (dy, dx) for directional drift - uncomment to enable

# Model parameters
GUIDANCE_SCALE=0.0
# device and random_seed are read from configs/frame1.yaml

# Run time-series generation
echo ""
echo "Starting time-series generation..."
echo "Checkpoint: $CHECKPOINT"
echo "Number of timepoints: $NUM_TIMEPOINTS"
echo "Displacement std: $DISPLACEMENT_STD pixels"
echo "Initial centres method: $METHOD"
if [ "$METHOD" == "from_file" ]; then
    echo "Centres file: $CENTRES_FILE"
fi
echo "Output directory: $OUTPUT_DIR"
echo ""

# Build command
CMD="python /users/kir-fritzsche/aif490/devel/tissue_analysis/CellSimul_diff/scripts/generate_timeseries.py \
    --checkpoint $CHECKPOINT \
    --num_timepoints $NUM_TIMEPOINTS \
    --displacement_std $DISPLACEMENT_STD \
    --displacement_mean $DISPLACEMENT_MEAN \
    --min_distance $MIN_DISTANCE \
    --noise_correlation $NOISE_CORRELATION \
    --temporal_smoothness $TEMPORAL_SMOOTHNESS \
    --output_dir $OUTPUT_DIR \
    --prefix $PREFIX \
    --method $METHOD \
    --guidance_scale $GUIDANCE_SCALE \
    --no_cfg"

# Add histogram matching if enabled
if [ "$MATCH_HISTOGRAMS" = "True" ]; then
    CMD="$CMD --match_histograms"
fi

# Add centres file if using from_file method
if [ "$METHOD" == "from_file" ]; then
    CMD="$CMD --centres_file $CENTRES_FILE"
fi

# Add density/min_distance if using poisson method
if [ "$METHOD" == "poisson" ]; then
    CMD="$CMD --density $DENSITY --min_distance $MIN_DISTANCE"
fi

# Add directional bias if specified
if [ ! -z "$DIRECTIONAL_BIAS" ]; then
    CMD="$CMD --directional_bias $DIRECTIONAL_BIAS"
fi

# Execute
eval $CMD

echo ""
echo "Time-series generation complete!"
echo "Output saved to: $OUTPUT_DIR"
