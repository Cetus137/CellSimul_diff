#!/bin/bash
#SBATCH --job-name=generate_cells_3d
#SBATCH --output=slogs/generate_3d_%j.out
#SBATCH --error=slogs/generate_3d_%j.err
#SBATCH --time=03:30:00
#SBATCH --partition=gpu_short
#SBATCH --gpus-per-node=1
#SBATCH --constraint="a100|rtx8000|v100"
#SBATCH --mem-per-gpu=10G
#SBATCH --cpus-per-task=2

# 3D Cell Diffusion Model - Synthetic Volume Generation
# Generates synthetic 3D cell microscopy volumes from trained model
# Centre methods: simple, poisson, training_dist, from_file, realistic
#   realistic   -> params from configs/frame1_3d.yaml (centre_generation_3d block)
#   poisson     -> uses DENSITY + MIN_DISTANCE
#   training_dist -> uses MIN_DISTANCE only


# Load required modules
module purge
source /well/kir/config/modules.sh
module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/12.0
source ~/devel/venv/Python-3.10.8-GCCcore-12.2.0/cell_simul_env/bin/activate

# Configuration
CHECKPOINT="/users/kir-fritzsche/aif490/devel/tissue_analysis/CellSimul_diff/checkpoints/unified_3d_node1_C_no_cfg_SNR/checkpoint_step_205000.pt"
NUM_SAMPLES=100
OUTPUT_DIR="/users/kir-fritzsche/aif490/devel/tissue_analysis/CellSimul_diff/synthetic_cells_3d/node1_only_C_no_cfg"
METHOD="realistic"  # Options: simple, poisson, training_dist, from_file, realistic
CENTRES_FILE="/users/kir-fritzsche/aif490/devel/tissue_analysis/CellSimul_diff/data_live_node1_3d/test/patch3d_f0005_00000_centres.npy"   # Only needed if METHOD=from_file

GUIDANCE_SCALE=0.0
DDIM_STEPS=0             # DDIM steps per volume; set to 0 to use full DDPM-1000
BATCH_SIZE=1             # Volumes per GPU pass — ~2x throughput; use 1 if VRAM OOM
CONFIG="/users/kir-fritzsche/aif490/devel/tissue_analysis/CellSimul_diff/configs/unified_3d.yaml"
DEVICE="cuda"            # "cuda" or "cpu"

# Run inference
echo ""
echo "Starting synthetic 3D cell generation..."
echo "Checkpoint: $CHECKPOINT"
echo "Number of samples: $NUM_SAMPLES"
echo "Method: $METHOD"
echo "Output directory: $OUTPUT_DIR"
echo ""

CMD="python /users/kir-fritzsche/aif490/devel/tissue_analysis/CellSimul_diff/scripts/generate_synthetic3d.py \
    --checkpoint \"$CHECKPOINT\" \
    --num_samples \"$NUM_SAMPLES\" \
    --output_dir \"$OUTPUT_DIR\" \
    --method \"$METHOD\" \
    --guidance_scale \"$GUIDANCE_SCALE\" \
    --ddim_steps \"$DDIM_STEPS\" \
    --config \"$CONFIG\" \
    --device \"$DEVICE\""

# density/min_distance: only used by poisson and training_dist methods
# (realistic reads these from the config's centre_generation_3d block instead)
if [ "$METHOD" == "poisson" ] || [ "$METHOD" == "training_dist" ]; then
    CMD="$CMD --min_distance \"$MIN_DISTANCE\""
fi
if [ "$METHOD" == "poisson" ]; then
    CMD="$CMD --density \"$DENSITY\""
fi

# Add centres file if using from_file method
if [ "$METHOD" == "from_file" ] && [ -n "$CENTRES_FILE" ]; then
    CMD="$CMD --centres_file \"$CENTRES_FILE\""
fi

# Disable CFG (remove --no_cfg to enable)

CMD="$CMD --no_cfg"

# Batch multiple volumes into a single DDPM pass for ~BATCH_SIZE x throughput
CMD="$CMD --batch_size \"$BATCH_SIZE\""

# Skip per-volume matplotlib PNGs (blocking CPU work); re-enable for debugging
CMD="$CMD --no_visualization"

# Skip saving _heatmap.tif conditioning files (saves disk I/O and storage)
#CMD="$CMD --no_heatmap"

eval $CMD

echo ""
echo "Generation complete!"
echo "Results saved to: $OUTPUT_DIR"
echo ""
