#!/bin/bash

#SBATCH --job-name      cell3d_preproc_array
#SBATCH --partition=short
##SBATCH --gpus-per-node=0
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time          00:05:00
#SBATCH --array=0-8000
#SBATCH --output        slogs/preprocess_3d/preprocess_3d_array.%A_%a.out
#SBATCH --error         slogs/preprocess_3d/preprocess_3d_array.%A_%a.err


# 3D Dataset Preprocessing - ARRAY JOB MODE
# Each task processes one cube file independently
# File index = $SLURM_ARRAY_TASK_ID
#
# IMPORTANT: Set --array=0-N where N = (number of cube files - 1)
# All tasks write directly to train/val/test folders with unique filenames
# No merge step needed!

echo "========================================"
echo "Cell Simulation 3D - Array Preprocessing"
echo "========================================"
echo "Job ID:    $SLURM_JOB_ID"
echo "Array ID:  $SLURM_ARRAY_JOB_ID"
echo "Task ID:   $SLURM_ARRAY_TASK_ID"
echo "Node:      $SLURM_NODELIST"
echo "CPUs:      $SLURM_CPUS_PER_TASK"
echo "Start:     $(date)"
echo "========================================"

WORK_DIR="/users/kir-fritzsche/aif490/devel/tissue_analysis/CellSimul_diff"
CONFIG="configs/frame1_3d.yaml"
FILE_INDEX=$SLURM_ARRAY_TASK_ID

cd $WORK_DIR

echo ""
echo "Loading environment..."
module purge
source /well/kir/config/modules.sh
module load Python/3.10.8-GCCcore-12.2.0
source ~/devel/venv/Python-3.10.8-GCCcore-12.2.0/cell_simul_env/bin/activate

echo ""
echo "Python: $(python --version)"
echo "Config: $CONFIG"
echo "Processing file index: $FILE_INDEX"
echo ""

python -c "
import sys
sys.path.insert(0, '.')
from preprocessing.build_dataset3d import build_dataset3d
build_dataset3d('$CONFIG', file_index=$FILE_INDEX)
"

EXIT_CODE=$?

echo ""
echo "========================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Preprocessing task $FILE_INDEX completed successfully!"
else
    echo "Preprocessing task $FILE_INDEX FAILED with exit code $EXIT_CODE"
fi
echo "End: $(date)"
echo "========================================"

exit $EXIT_CODE
