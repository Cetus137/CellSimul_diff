#!/bin/bash
#SBATCH --job-name=generate_two_frame
#SBATCH --partition=gpu_short
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-gpu=2G
#SBATCH --time          00:15:00
#SBATCH --output        slogs/generate_two_frame.%j.out
#SBATCH --error         slogs/generate_two_frame.%j.err
#SBATCH --exclude       compg009,compg010,compg011,compg013
##SBATCH --mem=32G

# Cell Simulation Diffusion Model - Two-Frame Generation
#
# Runs the temporally consistent two-frame sampling pipeline:
#
#   Step 1:  I_t     ~ p( I    | C_t )            [frame-1 model]
#   Step 2:  I_{t+1} ~ p( I    | C_{t+1}, I_t )  [frame-2 model]
#
# Prerequisites:
#   1. Trained frame-1 checkpoint  (checkpoints/long_training/best.pt)
#   2. Trained frame-2 checkpoint  (checkpoints/frame2/best.pt)
#   3. Centre .npy files for time t and t+1
#
# Run with:
#   sbatch SLURM/slurm_generate_two_frame.sl
#
# CPU fallback (slow but no GPU needed):
#   Change --partition=gpu_short  →  --partition=short
#   Remove or comment --gpus-per-node=1
#   Set BATCH_SIZE=1 below for manageable memory use

mkdir -p slogs

module purge
source /well/kir/config/modules.sh
module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/12.0
source ~/devel/venv/Python-3.10.8-GCCcore-12.2.0/cell_simul_env/bin/activate

WORK_DIR="/users/kir-fritzsche/aif490/devel/tissue_analysis/CellSimul_diff"
cd $WORK_DIR

# ---- Configuration  --------------------------------------------------
FRAME1_CKPT="/users/kir-fritzsche/aif490/devel/tissue_analysis/CellSimul_diff/checkpoints/frame1_multiz/best.pt"
FRAME2_CKPT="/users/kir-fritzsche/aif490/devel/tissue_analysis/CellSimul_diff/checkpoints/frame2/best.pt"
FRAME1_CONFIG="/users/kir-fritzsche/aif490/devel/tissue_analysis/CellSimul_diff/configs/frame1.yaml"
FRAME2_CONFIG="/users/kir-fritzsche/aif490/devel/tissue_analysis/CellSimul_diff/configs/frame2.yaml"

# Cell-centre files for time t and t+1
# Adjust these paths to your desired input centres.
CENTRES_T="/users/kir-fritzsche/aif490/devel/tissue_analysis/CellSimul_diff/timeseries_output/timeseries_t0000_centres.npy"
CENTRES_T1="/users/kir-fritzsche/aif490/devel/tissue_analysis/CellSimul_diff/timeseries_output/timeseries_t0005_centres.npy"

OUTPUT_DIR="/users/kir-fritzsche/aif490/devel/tissue_analysis/CellSimul_diff/two_frame_output/time"
IMAGE_SIZE=256
BATCH_SIZE=1

# Classifier-free guidance (set USE_CFG=true to enable)
USE_CFG=false
GUIDANCE_SCALE=3.0

# Load EMA weights from checkpoints (recommended for best quality)
USE_EMA=false

# Save concatenated (2, H, W) TIFF in addition to .npy files
SAVE_TIFF=true
# ----------------------------------------------------------------------

mkdir -p $OUTPUT_DIR

echo "========================================"
echo "Two-Frame Temporally Linked Generation"
echo "========================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $SLURM_NODELIST"
echo "GPU:           ${CUDA_VISIBLE_DEVICES:-none}"
echo "Start:         $(date)"
echo "========================================"
echo "Frame-1 ckpt:  $FRAME1_CKPT"
echo "Frame-2 ckpt:  $FRAME2_CKPT"
echo "Centres t:     $CENTRES_T"
echo "Centres t+1:   $CENTRES_T1"
echo "Output dir:    $OUTPUT_DIR"
echo "Batch size:    $BATCH_SIZE"
echo "CFG:           $USE_CFG  (scale=$GUIDANCE_SCALE)"
echo "EMA:           $USE_EMA"
echo "========================================"
echo ""

# Auto-detect device — falls back to CPU if no GPU is allocated.
# To run CPU-only, submit with: --partition=short (remove --gpus-per-node line)
if [ -n "$CUDA_VISIBLE_DEVICES" ] && [ "$CUDA_VISIBLE_DEVICES" != "NoDevFiles" ]; then
    DEVICE="cuda"
    echo "Device: CUDA (GPU $CUDA_VISIBLE_DEVICES)"
else
    DEVICE="cpu"
    echo "Device: CPU (no GPU allocated — generation will be slow)"
fi

# Build command
CMD="python -u -m sampling.sample_two_frame \
    --frame1_ckpt    $FRAME1_CKPT \
    --frame2_ckpt    $FRAME2_CKPT \
    --frame1_config  $FRAME1_CONFIG \
    --frame2_config  $FRAME2_CONFIG \
    --centres_t      $CENTRES_T \
    --centres_t1     $CENTRES_T1 \
    --out_dir        $OUTPUT_DIR \
    --image_size     $IMAGE_SIZE \
    --batch_size     $BATCH_SIZE \
    --guidance_scale $GUIDANCE_SCALE \
    --device         $DEVICE"

if [ "$USE_EMA" = "true" ]; then
    CMD="$CMD --use_ema"
fi

if [ "$SAVE_TIFF" = "true" ]; then
    CMD="$CMD --save_tiff"
fi

if [ "$USE_CFG" = "true" ]; then
    CMD="$CMD --cfg"
fi

echo "Starting generation..."
echo ""
eval $CMD

GEN_EXIT_CODE=$?

echo ""
echo "========================================"
if [ $GEN_EXIT_CODE -eq 0 ]; then
    echo "Generation completed successfully!"
    echo ""
    echo "Output files:"
    ls -lh $OUTPUT_DIR/ 2>/dev/null | head -20
    echo ""
    echo "To inspect results:"
    echo "  import numpy as np"
    echo "  I_t   = np.load('$OUTPUT_DIR/frame_t_sample000.npy')"
    echo "  I_t1  = np.load('$OUTPUT_DIR/frame_t1_sample000.npy')"
    echo "  # PNG previews also saved as pair_preview_sample*.png"
else
    echo "Generation FAILED  (exit code: $GEN_EXIT_CODE)"
fi
echo "End time: $(date)"
echo "========================================"

exit $GEN_EXIT_CODE
