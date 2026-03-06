#!/bin/bash
#SBATCH --job-name=img2img_3d
#SBATCH --output=slogs/img2img_3d_%j.out
#SBATCH --error=slogs/img2img_3d_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=gpu_short
#SBATCH --gpus-per-node=1
#SBATCH --constraint="a100|rtx8000|v100"
#SBATCH --mem-per-gpu=12G
#SBATCH --cpus-per-task=4

# SDEdit img2img temporal generation — no second model required.
#
# Takes real V_t0 + real C_t1 from PAIRS_DIR, generates a predicted V_t1
# by corrupting V_t0 to t_start then denoising conditioned on C_t1.
#
# t_start controls coherence vs. change:
#   ~100-150 : high fidelity to V_t0; only small local changes
#   ~250     : balanced — recommended first run
#   ~400-600 : large structural changes; loose coupling to V_t0

module purge
source /well/kir/config/modules.sh
module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/12.0
source ~/devel/venv/Python-3.10.8-GCCcore-12.2.0/cell_simul_env/bin/activate

WORK_DIR="/users/kir-fritzsche/aif490/devel/tissue_analysis/CellSimul_diff"
cd "$WORK_DIR"

# ── Configuration ──────────────────────────────────────────────────────────────
CHECKPOINT="$WORK_DIR/checkpoints/frame1_3d_combined_noD_noZ_raw/best.pt"
CONFIG="$WORK_DIR/configs/frame1_3d.yaml"

PAIRS_DIR="$WORK_DIR/data_live_node1_3d/pairs"

T_START=350          # Noise level (see header above)
USE_DDIM=false        # true = DDIM ~50 steps (fast); false = full DDPM 1000 steps
DDIM_STEPS=50        # Only used when USE_DDIM=true

BATCH_SIZE=1         # Pairs per GPU pass; reduce to 1 if VRAM OOM
MAX_PAIRS=50         # Limit for a test run; comment out or set "" for all pairs

SAVE_SRC=true        # Save source V_t0 (first frame)
SAVE_HEATMAP=true    # Save C_t1 conditioning heatmap
SAVE_REAL=true       # Save real V_t1 for comparison

OUTPUT_DIR="$WORK_DIR/synthetic_cells_3d/img2img/t_start_${T_START}"
# ──────────────────────────────────────────────────────────────────────────────

mkdir -p "$OUTPUT_DIR"
mkdir -p slogs

echo "================================================"
echo "SDEdit Img2Img 3D Generation"
echo "================================================"
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURM_NODELIST"
echo "GPU:        ${CUDA_VISIBLE_DEVICES:-none}"
echo "Start:      $(date)"
echo "t_start:    $T_START"
echo "DDIM:       $USE_DDIM  (steps=$DDIM_STEPS)"
echo "Pairs dir:  $PAIRS_DIR"
echo "Output:     $OUTPUT_DIR"
echo "================================================"
echo ""

# Determine device from SLURM GPU allocation
if [ -n "$CUDA_VISIBLE_DEVICES" ] && [ "$CUDA_VISIBLE_DEVICES" != "NoDevFiles" ]; then
    DEVICE="cuda"
else
    DEVICE="cpu"
fi

CMD="python $WORK_DIR/scripts/generate_timeseries3d_img2img.py \
    --checkpoint \"$CHECKPOINT\" \
    --config \"$CONFIG\" \
    --pairs_dir \"$PAIRS_DIR\" \
    --output_dir \"$OUTPUT_DIR\" \
    --t_start \"$T_START\" \
    --batch_size \"$BATCH_SIZE\" \
    --heatmap_sigma 3.0 \
    --device \"$DEVICE\" \
    --no_visualization"

if [ "$SAVE_SRC"     = true ]; then CMD="$CMD --save_src";     fi
if [ "$SAVE_HEATMAP" = true ]; then CMD="$CMD --save_heatmap"; fi
if [ "$SAVE_REAL"    = true ]; then CMD="$CMD --save_real";    fi

if [ -n "$MAX_PAIRS" ]; then
    CMD="$CMD --max_pairs \"$MAX_PAIRS\""
fi

if [ "$USE_DDIM" = true ]; then
    CMD="$CMD --use_ddim --ddim_steps \"$DDIM_STEPS\""
fi

eval $CMD

echo ""
echo "Generation complete!"
echo "Results saved to: $OUTPUT_DIR"
echo ""
