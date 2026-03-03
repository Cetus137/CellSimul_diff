#!/bin/bash

#SBATCH --job-name      centre_stats_3d
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time          00:30:00
#SBATCH --output        slogs/centre_stats_3d.%j.out
#SBATCH --error         slogs/centre_stats_3d.%j.err


# 3D Centre Statistics Extraction
#
# Scans all *_centres.npy files in the 3D training splits (both nodes),
# computes:
#   - Cell count N distribution (mean, std, percentiles)
#   - Per-cell nearest-neighbour distance distribution
#   - 16-bin marginal KDE grids for z, y, x axes
#
# Writes a ready-to-paste YAML block to STATS_OUTPUT, which should be
# copied into configs/frame1_3d.yaml under the `centre_generation_3d` key.
#
# USAGE:
#   sbatch SLURM/slurm_analyse_centres_3d.sl

WORK_DIR="/users/kir-fritzsche/aif490/devel/tissue_analysis/CellSimul_diff"

# Directories to scan (space-separated; both train splits)
PATCHES_DIRS=(
    "data_live_node1_3d/train"
    "data_live_node2_3d/train"
)

# Where to write the output YAML block
STATS_OUTPUT="centre_stats_3d.yaml"

# Number of histogram bins for marginal KDE grids
N_BINS=16

# Optional: cap the number of patches processed (comment out for full scan)
# MAX_PATCHES=500

cd "$WORK_DIR"

# Load environment
module purge
source /well/kir/config/modules.sh
module load Python/3.10.8-GCCcore-12.2.0
source ~/devel/venv/Python-3.10.8-GCCcore-12.2.0/cell_simul_env/bin/activate

echo "========================================"
echo "Cell Simulation 3D - Centre Statistics"
echo "========================================"
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURM_NODELIST"
echo "CPUs:     $SLURM_CPUS_PER_TASK"
echo "Start:    $(date)"
echo "========================================"
echo ""
echo "Python: $(python --version)"
echo "Work dir: $WORK_DIR"
echo "Patches dirs: ${PATCHES_DIRS[*]}"
echo "Output YAML: $STATS_OUTPUT"
echo ""

CMD="python scripts/analyze_training_stats.py \
    --patches_dir ${PATCHES_DIRS[*]} \
    --save_stats \"$STATS_OUTPUT\" \
    --n_bins $N_BINS"

# Uncomment to cap the number of patches (faster for a quick check):
# CMD="$CMD --max_patches $MAX_PATCHES"

echo "Running: $CMD"
echo ""
eval $CMD
STATUS=$?

echo ""
echo "========================================"
if [ $STATUS -eq 0 ]; then
    echo "Stats extraction complete!"
    echo "Output written to: $WORK_DIR/$STATS_OUTPUT"
    echo ""
    echo "Next steps:"
    echo "  1. Review the printed distributions above"
    echo "  2. Paste the 'centre_generation_3d' block from $STATS_OUTPUT"
    echo "     into configs/frame1_3d.yaml"
    echo "  3. Re-run generation with --method realistic"
else
    echo "ERROR: Stats extraction failed (exit code $STATUS)"
fi
echo "End: $(date)"
echo "========================================"

exit $STATUS
