#!/bin/bash
#SBATCH --job-name=build_temporal
#SBATCH --output=slogs/build_temporal_%j.out
#SBATCH --error=slogs/build_temporal_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

# Cell Simulation Diffusion Model - Temporal Pair Dataset Builder
# Scans data_multiz/raw/ and data_multiz/segmented/ for consecutive-T tile pairs,
# extracts cell centres from segmentation masks, normalises raw images to [-1,1],
# and saves per-pair .npy arrays to data_multiz/processed/pairs/{train,val,test}/
#
# Split is 80/10/10 on unique spatial positions (C, z, y, x) to prevent
# the same region appearing in both train and val/test.
#
# Run with:
#   sbatch SLURM/slurm_build_temporal_dataset.sl

# Load environment
module purge
source /well/kir/config/modules.sh
module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/12.0
source ~/devel/venv/Python-3.10.8-GCCcore-12.2.0/cell_simul_env/bin/activate

WORK_DIR="/users/kir-fritzsche/aif490/devel/tissue_analysis/CellSimul_diff"
cd $WORK_DIR

# Paths, split fractions, min_cells, and random_seed are read from configs/frame2.yaml
# (temporal_dataset section).  Override any value with a CLI flag if needed.

echo "========================================"
echo "Building Temporal Pair Dataset"
echo "========================================"
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURM_NODELIST"
echo "CPUs:      $SLURM_CPUS_PER_TASK"
echo "Start:     $(date)"
echo "========================================"

echo ""
echo "Python version:"
python --version

echo ""
echo "Starting dataset build..."
echo ""

python -u -m preprocessing.build_temporal_dataset \
    --config configs/frame2.yaml

BUILD_EXIT_CODE=$?

echo ""
echo "========================================"
if [ $BUILD_EXIT_CODE -eq 0 ]; then
    echo "Dataset build completed successfully!"
else
    echo "Dataset build FAILED  (exit code: $BUILD_EXIT_CODE)"
fi
echo "End time: $(date)"
echo "========================================"

exit $BUILD_EXIT_CODE
