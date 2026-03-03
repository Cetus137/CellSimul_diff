#!/bin/bash

#SBATCH --job-name      cell3d_preprocess
#SBATCH --partition=short
##SBATCH --gpus-per-node=0
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time          04:00:00
#SBATCH --output        slogs/preprocess_3d.%j.out
#SBATCH --error         slogs/preprocess_3d.%j.err


# 3D Dataset Preprocessing
# 
# USAGE:
#   Normal mode:  sbatch slurm_preprocess_3d.sl
#   Array mode:   sbatch --array=0-N slurm_preprocess_3d.sl  (where N = num_files - 1)
#
# Array mode: Each task processes one file and writes directly to train/val/test
#             based on deterministic split assignment (no merge needed!)

WORK_DIR="/users/kir-fritzsche/aif490/devel/tissue_analysis/CellSimul_diff"
CONFIG="configs/frame1_3d.yaml"

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
    echo "Cell Simulation 3D - Array Processing"
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
    echo "Processing file index: $SLURM_ARRAY_TASK_ID"
    echo ""
    
    python -c "
import sys
sys.path.insert(0, '.')
from preprocessing.build_dataset3d import build_dataset3d
build_dataset3d('$CONFIG', file_index=$SLURM_ARRAY_TASK_ID)
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
    echo "Cell Simulation 3D - Preprocessing"
    echo "========================================"
    echo "Job ID:  $SLURM_JOB_ID"
    echo "Node:    $SLURM_NODELIST"
    echo "CPUs:    $SLURM_CPUS_PER_TASK"
    echo "Start:   $(date)"
    echo "========================================"
    echo ""
    echo "Python: $(python --version)"
    echo "Config: $CONFIG"
    echo ""
    
    python -c "
import sys
sys.path.insert(0, '.')
from preprocessing.build_dataset3d import build_dataset3d
build_dataset3d('$CONFIG')
"
    
    EXIT_CODE=$?
    
    echo ""
    echo "========================================"
    if [ $EXIT_CODE -eq 0 ]; then
        echo "Preprocessing completed successfully!"
    else
        echo "Preprocessing FAILED with exit code $EXIT_CODE"
    fi
    echo "End: $(date)"
    echo "========================================"
    
    exit $EXIT_CODE
fi
