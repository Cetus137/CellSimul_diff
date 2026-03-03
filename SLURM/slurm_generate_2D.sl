#!/bin/bash
#SBATCH --job-name=generate_cells
#SBATCH --output=slogs/generate_%j.out
#SBATCH --error=slogs/generate_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=gpu_short
#SBATCH --gpus-per-node=1
#SBATCH --constraint="a100|rtx8000|v100"
#SBATCH --mem-per-gpu=2G

# Cell Diffusion Model - Synthetic Image Generation
# Generates synthetic cell microscopy images from trained model
# Uses Poisson disk sampling for realistic cell centre distributions


# Load required modules (adjust for your HPC)
module purge
source /well/kir/config/modules.sh
module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/12.0
source ~/devel/venv/Python-3.10.8-GCCcore-12.2.0/cell_simul_env/bin/activate

# Configuration
CHECKPOINT="/users/kir-fritzsche/aif490/devel/tissue_analysis/CellSimul_diff/checkpoints/frame1_multiz/best.pt"
NUM_SAMPLES=10
OUTPUT_DIR="/users/kir-fritzsche/aif490/devel/tissue_analysis/CellSimul_diff/synthetic_cells"
METHOD="from_file"  # Options: simple, poisson, training_dist, from_file
CENTRES_FILE="/users/kir-fritzsche/aif490/devel/tissue_analysis/CellSimul_diff/data/processed/test/patch_00000_centres.npy"
DENSITY=0.0003    # Cells per pixel (adjust based on your data)
MIN_DISTANCE=24.0 # Minimum spacing between cells
GUIDANCE_SCALE=0.0
CONFIG="/users/kir-fritzsche/aif490/devel/tissue_analysis/CellSimul_diff/configs/frame1.yaml"
# device and random_seed are read from configs/frame1.yaml

# Run inference
echo ""
echo "Starting synthetic cell generation..."
echo "Checkpoint: $CHECKPOINT"
echo "Number of samples: $NUM_SAMPLES"
echo "Method: $METHOD"
echo "Output directory: $OUTPUT_DIR"
echo ""

python /users/kir-fritzsche/aif490/devel/tissue_analysis/CellSimul_diff/scripts/generate_synthetic.py \
    --checkpoint "$CHECKPOINT" \
    --num_samples "$NUM_SAMPLES" \
    --output_dir "$OUTPUT_DIR" \
    --method "$METHOD" \
    --centres_file "$CENTRES_FILE" \
    --density "$DENSITY" \
    --min_distance "$MIN_DISTANCE" \
    --guidance_scale "$GUIDANCE_SCALE" \
    --no_cfg \
    --config "$CONFIG"


echo ""
echo "Generation complete!"
echo "Results saved to: $OUTPUT_DIR"
echo ""
