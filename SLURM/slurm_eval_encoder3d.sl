#!/bin/bash
#SBATCH --job-name=eval_encoder_3d
#SBATCH --partition=gpu_short
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-gpu=16G
#SBATCH --time=00:45:00
#SBATCH --output=slogs/eval_encoder_3d.%j.out
#SBATCH --error=slogs/eval_encoder_3d.%j.err
#SBATCH --exclude=compg009,compg010,compg011,compg013

# Embedding-based realism evaluation (GPU required)
#   Frozen frame1_3d UNet encoder → latent space comparison
#   Fréchet Distance, linear probe balanced accuracy, t-SNE plot
#
# Outputs (in EVAL_OUT_DIR):
#   evaluation_summary.yaml
#   tsne_embeddings.png
#   embeddings_real.npy
#   embeddings_syn.npy
#
# Run with:  sbatch SLURM/slurm_eval_encoder3d.sl

module purge
source /well/kir/config/modules.sh
module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/12.0
source ~/devel/venv/Python-3.10.8-GCCcore-12.2.0/cell_simul_env/bin/activate

WORK_DIR="/users/kir-fritzsche/aif490/devel/tissue_analysis/CellSimul_diff"
cd $WORK_DIR

# ── Configuration ──────────────────────────────────────────────────────────────
REAL_DIR1="/users/kir-fritzsche/aif490/devel/tissue_analysis/CellSimul_diff/data_live_node1_3d/test"

SYN_DIR="/users/kir-fritzsche/aif490/devel/tissue_analysis/CellSimul_diff/synthetic_cells_3d/noZ_raw/from_test2"  # Must match the output dir used in slurm_generate_3d.sl

CKPT="/users/kir-fritzsche/aif490/devel/tissue_analysis/CellSimul_diff/checkpoints/frame1_3d_combined_noD_noZ_raw/best.pt"
CONFIG="/users/kir-fritzsche/aif490/devel/tissue_analysis/CellSimul_diff/configs/frame1_3d.yaml"

EVAL_OUT_DIR="/users/kir-fritzsche/aif490/devel/tissue_analysis/CellSimul_diff/evaluation/results_noZ_raw/from_test/encoder"

# Volumes per encoder forward pass — reduce to 2 if GPU OOM
BATCH_SIZE=4

# Heatmap sigma — must match the value used during training
HEATMAP_SIGMA=3.0

MAX_REAL=""   # e.g. MAX_REAL=200
MAX_SYN=""    # e.g. MAX_SYN=50
# ──────────────────────────────────────────────────────────────────────────────

mkdir -p "$EVAL_OUT_DIR"
mkdir -p slogs

echo "================================================"
echo "Encoder-based 3D Realism Evaluation"
echo "================================================"
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURM_NODELIST"
echo "GPU:        ${CUDA_VISIBLE_DEVICES:-none}"
echo "Start:      $(date)"
echo "Real dir:   $REAL_DIR1"
echo "Syn dir:    $SYN_DIR"
echo "Checkpoint: $CKPT"
echo "Out dir:    $EVAL_OUT_DIR"
echo "Batch size: $BATCH_SIZE"
echo "================================================"
echo ""

if [ -n "$CUDA_VISIBLE_DEVICES" ] && [ "$CUDA_VISIBLE_DEVICES" != "NoDevFiles" ]; then
    DEVICE="cuda"
    echo "Device: CUDA (GPU $CUDA_VISIBLE_DEVICES)"
else
    DEVICE="cpu"
    echo "Device: CPU (no GPU allocated — encoder will be slow)"
fi

python --version

CMD="python -u -m scripts.evaluate_realism3d \
    --real_dirs        \"$REAL_DIR1\" \
    --syn_dir          \"$SYN_DIR\" \
    --ckpt             \"$CKPT\" \
    --config           \"$CONFIG\" \
    --out_dir          \"$EVAL_OUT_DIR\" \
    --batch_size       $BATCH_SIZE \
    --heatmap_sigma    $HEATMAP_SIGMA \
    --device           $DEVICE \
    --no_pixel"

if [ -n "$MAX_REAL" ]; then CMD="$CMD --max_real $MAX_REAL"; fi
if [ -n "$MAX_SYN"  ]; then CMD="$CMD --max_syn  $MAX_SYN";  fi

eval $CMD
EXIT_CODE=$?

echo ""
echo "================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Encoder evaluation complete."
    ls -lh "$EVAL_OUT_DIR"
    echo ""
    cat "$EVAL_OUT_DIR/evaluation_summary.yaml" 2>/dev/null
else
    echo "FAILED (exit $EXIT_CODE)"
fi
echo "End: $(date)"
echo "================================================"
exit $EXIT_CODE
