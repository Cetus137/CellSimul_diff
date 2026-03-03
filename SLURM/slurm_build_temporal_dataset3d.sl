#!/bin/bash

#SBATCH --job-name      build_temporal_3d
#SBATCH --partition=short
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time          00:02:00
#SBATCH --output        slogs/build_temporal/build_temporal_3d.%j.out
#SBATCH --error         slogs/build_temporal/build_temporal_3d.%j.err


# 3D Temporal Pair Dataset Builder
#
# USAGE:
#   Normal mode:  sbatch SLURM/slurm_build_temporal_dataset3d.sl
#   Array mode:   sbatch --array=0-N SLURM/slurm_build_temporal_dataset3d.sl
#                 (where N = num_pairs - 1, e.g., --array=0-1599 for 1600 pairs)
#
# Array mode: Each task processes one temporal pair and writes patches directly
#             to train/val/test based on deterministic split assignment.
#             No merge needed!
#
# To determine N: Run normal mode briefly (it will print "Found X pairs" early),
#                 then cancel and resubmit with --array=0-$(X-1)

WORK_DIR="/users/kir-fritzsche/aif490/devel/tissue_analysis/CellSimul_diff"
CONFIG="configs/frame2_3d.yaml"

cd $WORK_DIR

# Load environment
module purge
source /well/kir/config/modules.sh
module load Python/3.10.8-GCCcore-12.2.0
source ~/devel/venv/Python-3.10.8-GCCcore-12.2.0/cell_simul_env/bin/activate


# ──────────────────────────────────────────────────────────────────────────────
# Check if running in array mode
# ──────────────────────────────────────────────────────────────────────────────

if [ ! -z "$SLURM_ARRAY_TASK_ID" ]; then
    # ═════════════ ARRAY MODE ═════════════
    echo "========================================"
    echo "3D Temporal Pair - Array Processing"
    echo "========================================"
    echo "Job ID:    $SLURM_JOB_ID"
    echo "Array ID:  $SLURM_ARRAY_JOB_ID"
    echo "Task ID:   $SLURM_ARRAY_TASK_ID"
    echo "Node:      $SLURM_NODELIST"
    echo "CPUs:      $SLURM_CPUS_PER_TASK"
    echo "Start:     $(date)"
    echo "========================================"
    echo ""
    echo "Python: $(python --version)"
    echo "Config: $CONFIG"
    echo "Processing pair index: $SLURM_ARRAY_TASK_ID"
    echo ""
    
    python -c "
import sys
sys.path.insert(0, '.')
from preprocessing.build_temporal_dataset3d import build_temporal_dataset3d
build_temporal_dataset3d('$CONFIG', pair_index=$SLURM_ARRAY_TASK_ID)
"
    
    EXIT_CODE=$?
    
    echo ""
    echo "========================================"
    if [ $EXIT_CODE -eq 0 ]; then
        echo "Task $SLURM_ARRAY_TASK_ID completed successfully!"
    else
        echo "Task $SLURM_ARRAY_TASK_ID FAILED with exit code $EXIT_CODE"
    fi
    echo "End: $(date)"
    echo "========================================"
    
    exit $EXIT_CODE

else
    # ═════════════ NORMAL MODE ═════════════
    echo "========================================"
    echo "Building 3D Temporal Pair Dataset"
    echo "========================================"
    echo "Job ID:    $SLURM_JOB_ID"
    echo "Node:      $SLURM_NODELIST"
    echo "CPUs:      $SLURM_CPUS_PER_TASK"
    echo "Memory:    ${SLURM_MEM_PER_NODE}MB"
    echo "Start:     $(date)"
    echo "========================================"
    echo ""
    echo "Python: $(python --version)"
    echo "Config: $CONFIG"
    echo ""
    
    python -c "
import sys
sys.path.insert(0, '.')
from preprocessing.build_temporal_dataset3d import build_temporal_dataset3d
build_temporal_dataset3d('$CONFIG')
"
    
    BUILD_EXIT_CODE=$?
    
    echo ""
    echo "========================================"
    if [ $BUILD_EXIT_CODE -eq 0 ]; then
        echo "3D Temporal dataset build completed successfully!"
        echo ""
        echo "Output directory: data_live_node1_3d/pairs/"
        echo "  train/ val/ test/ subdirectories created"
        echo ""
        echo "Next steps:"
        echo "  1. Train frame2_3d model:"
        echo "     sbatch SLURM/slurm_train_frame2_3d.sl"
        echo "  2. Or run debug overfit test first:"
        echo "     python -m scripts.train_frame2_3d --config configs/frame2_3d.yaml --debug_overfit"
    else
        echo "Dataset build FAILED  (exit code: $BUILD_EXIT_CODE)"
        echo "Check the log above for errors."
    fi
    echo "End time: $(date)"
    echo "========================================"
    
    exit $BUILD_EXIT_CODE
fi
