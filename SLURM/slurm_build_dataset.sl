#!/bin/bash
#SBATCH --job-name=preprocess_cells
#SBATCH --output=slogs/build_dataset%j.out
#SBATCH --error=slogs/build_dataset%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G

# Cell Simulation Diffusion Model - Frame-1 Data Preprocessing
# Reads raw images + masks from data_multiz/, extracts 256×256 patches,
# filters by min_cells_per_patch (set in the config YAML), and writes
# directly into train/val/test subdirectories under PATCHES_DIR.
# A skip summary (images without masks/centres, patches below min_cells)
# is printed to the .err log at the end of the run.
#
# Run with:
#   sbatch SLURM/slurm_preprocess.sl

# Load environment
module purge
source /well/kir/config/modules.sh
module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/12.0
source ~/devel/venv/Python-3.10.8-GCCcore-12.2.0/cell_simul_env/bin/activate

WORK_DIR="/users/kir-fritzsche/aif490/devel/tissue_analysis/CellSimul_diff"
cd $WORK_DIR

# ---- Configuration ---------------------------------------------------
# All paths and thresholds (including min_cells_per_patch) are
# controlled from the YAML config — edit there, not here.
CONFIG="configs/frame1.yaml"

# ----------------------------------------------------------------------

mkdir -p slogs

echo "========================================"
echo "Cell Diffusion - Data Preprocessing"
echo "========================================"
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURM_NODELIST"
echo "CPUs:      $SLURM_CPUS_PER_TASK"
echo "Config:    $CONFIG"
echo "Start:     $(date)"
echo "========================================"

echo ""
echo "Python version:"
python --version

echo ""
echo "Key config settings:"
grep -E "min_cells_per_patch|patch_size|patch_stride|images_dir|masks_dir" $CONFIG || true

echo ""
echo "Starting preprocessing pipeline..."
echo ""

python -u scripts/preprocess_data.py \
    --config $CONFIG \
    --force

PREPROC_EXIT_CODE=$?

echo ""
echo "========================================"
echo "End time: $(date)"
echo "========================================"

exit $PREPROC_EXIT_CODE
