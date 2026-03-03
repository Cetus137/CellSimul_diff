#!/bin/bash
#SBATCH --job-name=eval_pixel_3d
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=slogs/eval_pixel_3d.%j.out
#SBATCH --error=slogs/eval_pixel_3d.%j.err

# Pixel-level realism evaluation (no GPU needed)
#   SNR, 3D radial power spectrum, intensity histograms
#
# Outputs (in EVAL_OUT_DIR):
#   evaluation_summary.yaml
#   power_spectrum_3d.png
#   intensity_histogram_3d.png
#
# Run with:  sbatch SLURM/slurm_eval_pixel3d.sl

module purge
source /well/kir/config/modules.sh
module load Python/3.10.8-GCCcore-12.2.0
source ~/devel/venv/Python-3.10.8-GCCcore-12.2.0/cell_simul_env/bin/activate

WORK_DIR="/users/kir-fritzsche/aif490/devel/tissue_analysis/CellSimul_diff"
cd $WORK_DIR

# ── Configuration ──────────────────────────────────────────────────────────────
REAL_DIR1="/users/kir-fritzsche/aif490/devel/tissue_analysis/CellSimul_diff/data_live_node1_3d/test"

SYN_DIR="/users/kir-fritzsche/aif490/devel/tissue_analysis/CellSimul_diff/data_synthetic_3D_frame1_volumes/no_z_raw"

# Dummy ckpt/config values — not used when --no_encoder is set, but required by argparse
CKPT="/users/kir-fritzsche/aif490/devel/tissue_analysis/CellSimul_diff/checkpoints/frame1_3d_combined_noD_noZ_raw/best.pt"
CONFIG="/users/kir-fritzsche/aif490/devel/tissue_analysis/CellSimul_diff/configs/frame1_3d.yaml"

EVAL_OUT_DIR="/users/kir-fritzsche/aif490/devel/tissue_analysis/CellSimul_diff/evaluation/results_noZ_raw/pixel"

MAX_REAL=""   # e.g. MAX_REAL=200
MAX_SYN=""    # e.g. MAX_SYN=50
# ──────────────────────────────────────────────────────────────────────────────

mkdir -p "$EVAL_OUT_DIR"
mkdir -p slogs

echo "================================================"
echo "Pixel-level 3D Realism Evaluation"
echo "================================================"
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURM_NODELIST"
echo "Start:     $(date)"
echo "Real dir:  $REAL_DIR1"
echo "Syn dir:   $SYN_DIR"
echo "Out dir:   $EVAL_OUT_DIR"
echo "================================================"
echo ""

python --version

CMD="python -u -m scripts.evaluate_realism3d \
    --real_dirs \"$REAL_DIR1\" \
    --syn_dir   \"$SYN_DIR\" \
    --ckpt      \"$CKPT\" \
    --config    \"$CONFIG\" \
    --out_dir   \"$EVAL_OUT_DIR\" \
    --no_encoder \
    --device cpu"

if [ -n "$MAX_REAL" ]; then CMD="$CMD --max_real $MAX_REAL"; fi
if [ -n "$MAX_SYN"  ]; then CMD="$CMD --max_syn  $MAX_SYN";  fi

eval $CMD
EXIT_CODE=$?

echo ""
echo "================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Pixel evaluation complete."
    ls -lh "$EVAL_OUT_DIR"
    echo ""
    cat "$EVAL_OUT_DIR/evaluation_summary.yaml" 2>/dev/null
else
    echo "FAILED (exit $EXIT_CODE)"
fi
echo "End: $(date)"
echo "================================================"
exit $EXIT_CODE
